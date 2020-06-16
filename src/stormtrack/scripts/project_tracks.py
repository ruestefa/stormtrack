#!/usr/bin/env python3

# Standard library
import argparse
import os
import sys
from functools import partial
from multiprocessing import Manager
from multiprocessing import Pool

# Third-party
import netCDF4 as nc4
import numpy as np

# Local
from ..core.io import read_feature_file
from ..utils.netcdf import nc_prepare_file
from ..utils.various import TimestepGenerator
from ..utils.various import TimestepStringFormatter
from ..utils.various import extract_args
from ..utils.various import print_args


def main(tss__infile__ts_outfile_lst__lst, feature_name, conf_lonlat, conf_out):

    # Read lon, lat
    lon, lat = read_lonlat_1d(**conf_lonlat)
    nx, ny = len(lon), len(lat)
    print(f"nx, ny = {nx}, {ny}")

    for timesteps, infile, ts_outfile_lst in tss__infile__ts_outfile_lst__lst:

        # Read tracks from file
        result = read_feature_file(
            infile,
            feature_name=feature_name,
            rebuild_pixels_if_necessary=True,
            counter=True,
            silent=False,
        )

        # Sort tracks by timesteps
        tracks_by_ts = {}
        for track in result["tracks"]:
            for timestep in track.timesteps():
                if timestep not in tracks_by_ts:
                    tracks_by_ts[timestep] = []
                tracks_by_ts[timestep].append(track)

        # Compute and write track property fields at each timestep
        for timestep, outfile in ts_outfile_lst:
            tracks = tracks_by_ts.get(timestep, [])

            print(f"[{timestep}] {len(tracks)} track{'s' if len(tracks) == 1 else 's'}")

            # Project track properties onto field by aid of masks
            flds = {
                "duration": np.zeros([nx, ny], np.int32),
                "age": np.zeros([nx, ny], np.int32),
            }
            for track in tracks:
                mask_i = track.to_mask(nx, ny, timesteps=[timestep])
                flds["duration"][mask_i] = track.duration()
                flds["age"][mask_i] = track.age(timestep)

            # Write fields to disk
            print(f"[{timestep}] write {len(flds)} fields to {outfile}")
            write_fields(outfile, flds, lon=lon, lat=lat, timestep=timestep, **conf_out)


def read_lonlat_1d(infile, name_lon="lon", name_lat="lat", transpose=False):

    print(f"read {name_lon}, {name_lat} from {infile}")

    # Read fields
    try:
        with nc4.Dataset(infile, "r") as fi:
            lon = fi[name_lon][:]
            lat = fi[name_lat][:]
    except OSError as e:
        raise OSError(f"{e} ({infile})")

    # Remove any leading dummy dimensions
    if lon.shape != lat.shape:
        raise Exception(f"lon and lat differ in shape: {lon.shape} != {lat.shape}")
    while len(lon.shape) > 2:
        if lon.shape[0] != 1:
            raise Exception(f"leading dimension > 1: {lon.shape}")
        lon = lon[0]
        lat = lat[0]

    if transpose:
        lon, lat = lon.T, lat.T

    if len(lon.shape) == 2:
        if (lon.mean(axis=1) != lon[:, 0]).all():
            raise Exception(f"2D lon not stacked 1D: {lon}")
        lon = lon[:, 0]

    if len(lat.shape) == 2:
        if (lat.mean(axis=0) != lat[0, :]).all():
            raise Exception(f"2D lat not stacked 1D: {lat}")
        lat = lat[0, :]

    return lon, lat


def write_fields(
    outfile,
    flds,
    *,
    lon=None,
    lat=None,
    timestep=None,
    overwrite_files=False,
    append_to_files=False,
    transpose=False,
    prefix=None,
    overwrite_vars=False,
):

    if overwrite_files and append_to_files:
        err = "overwrite_files and append_to_files cannot both be True"
        raise ValueError(err)

    kwas_append = dict(
        transpose=transpose, prefix=prefix, overwrite_vars=overwrite_vars,
    )

    kwas_create = dict(
        transpose=transpose, prefix=prefix, lon=lon, lat=lat, timestep=timestep,
    )

    if os.path.exists(outfile):

        if append_to_files:
            with nc4.Dataset(outfile, "a") as fa:
                _write_fields__append(fa, flds, **kwas_append)

        elif overwrite_files:
            _write_fields__create(outfile, flds, **kwas_create)

        else:
            raise Exception(
                "file already exists: {outfile}"
                "(pass --overwrite-files or --append-to-files)"
            )
    else:
        _write_fields__create(outfile, flds, **kwas_create)


def _write_fields__create(outfile, flds, *, lon, lat, timestep, **kwargs):

    if lon is None:
        raise ValueError("must pass 'lon' to create new file")
    if lat is None:
        raise ValueError("must pass 'lat' to create new file")
    if timestep is None:
        raise ValueError("must pass 'timestep' to create new file")

    dims = ["time", "lat", "lon"]

    with nc4.Dataset(outfile, "w") as fo:
        nc_prepare_file(fo, dims, lat=lat, lon=lon, timesteps=[timestep])
        _write_fields__append(fo, flds, **kwargs)


def _write_fields__append(fo, flds, *, transpose, prefix, overwrite_vars=False):

    # Determine field dimensions
    dims = []
    if "time" in fo.dimensions:
        dims.append("time")
    if "lat" in fo.dimensions and "lon" in fo.dimensions:
        dims.extend(["lat", "lon"])
    elif "rlat" in fo.dimensions and "rlon" in fo.dimensions:
        dims.extend(["rlat", "rlon"])
    else:
        err = "neither (lat, lon) nor (rlat, rlon) in dimensions"
        raise Exception(err)

    # Add fields
    for name, fld in flds.items():

        if transpose:
            fld = fld.T

        if len(dims) == 3:
            fld = np.expand_dims(fld, 0)

        if prefix:
            name = f"{prefix}_{name}"

        if name in fo.variables:
            if not overwrite_vars:
                raise Exception(
                    f"variable '{name}' already exists in file {fo.name} "
                    "(to overwrite, pass --overwrite-vars)"
                )
            var = fo.variables[name]
        else:
            var = fo.createVariable(name, fld.dtype, dims)

        var.grid_mapping = "rotated_pole"
        var.coordinates = "lon lat"

        var[:] = fld


def setup_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    grp = parser.add_argument_group("input")
    grp.add_argument(
        "-i",
        "--infile-fmt",
        help="feature track file format (with {YYYY} etc.)",
        metavar="format-str",
        dest="infile_fmt",
        required=True,
    )
    grp.add_argument(
        "-s",
        "--timesteps",
        help="timesteps used to reconstruce input file names",
        metavar="YYYYMMDDHH",
        nargs="+",
        type=int,
        dest="tss",
    )
    grp.add_argument(
        "-S",
        "--timesteps-range",
        help="timestep range (format: YYYYMMDDHH; stride in hours)",
        metavar=("start", "end", "stride"),
        nargs=3,
        action="append",
        type=int,
        dest="ts_ranges",
    )
    grp.add_argument(
        "-f",
        "--feature-name",
        help="feature name",
        metavar="name",
        dest="feature_name",
        required=True,
    )
    grp = parser.add_argument_group("grid")
    grp.add_argument(
        "--lonlat-file",
        help="input file with lon/lat fields",
        metavar="filename",
        dest="lonlat__infile",
        required=True,
    )
    grp.add_argument(
        "--lonlat-names",
        help="variable names of lon/lat fields",
        nargs=2,
        default=("lon", "lat"),
        dest="lonlat__varnames",
    )
    grp.add_argument(
        "--lonlat-transpose",
        help="transpose lon/lat fields",
        action="store_true",
        dest="lonlat__transpose",
    )
    grp = parser.add_argument_group("output")
    grp.add_argument(
        "-o",
        "--outfile-fmt",
        help=(
            "output file format string (with {YYYY} etc.); "
            "only one timestep per output file supported"
        ),
        metavar="file-fmt",
        dest="outfile_fmt",
        required=True,
    )
    grp.add_argument(
        "-p",
        "--outfield-prefix",
        help="first part of name of output fields",
        metavar="str",
        dest="out__prefix",
    )
    grp.add_argument(
        "--overwrite-files",
        help="overwrite existing files",
        action="store_true",
        dest="out__overwrite_files",
    )
    grp.add_argument(
        "--append-to-files",
        help="append fields to existing files",
        action="store_true",
        dest="out__append_to_files",
    )
    grp.add_argument(
        "--overwrite-vars",
        help="overwrite existing variables",
        action="store_true",
        dest="out__overwrite_vars",
    )
    grp.add_argument(
        "--transpose-fields",
        help="transpose fields before writing them to disk",
        action="store_true",
        default=False,
        dest="out__transpose",
    )
    grp.add_argument(
        "--notranspose-fields",
        help="don't transpose fields before writing them to disk",
        action="store_false",
        default=False,
        dest="out__transpose",
    )
    return parser


def preproc_args(parser, kwargs):

    for key in ["lonlat", "out"]:
        kwargs["conf_" + key] = extract_args(kwargs, key)

    # -- Input/output

    # Prepare timesteps
    try:
        timesteps = TimestepGenerator.from_args(
            kwargs.pop("tss"), kwargs.pop("ts_ranges")
        )
    except ValueError as e:
        parser.error("error preparing timesteps: " + str(e))

    # Construct input files
    infile_fmt = kwargs.pop("infile_fmt")
    infiles_by_tss = TimestepStringFormatter(infile_fmt).run(timesteps)

    # Construct output files
    outfile_fmt = kwargs.pop("outfile_fmt")
    outfiles_by_tss = TimestepStringFormatter(outfile_fmt).run(timesteps)
    if any(len(tss) != 1 for tss in outfiles_by_tss.keys()):
        parser.error(
            "invalid output file (each must contain one timestep): " + outfile_fmt
        )

    # Collect output files for each input file
    _lst = []
    for tss, infile in sorted(infiles_by_tss.items()):
        ts_outfile_lst = []
        for (ts,), outfile in sorted(outfiles_by_tss.items()):
            if ts not in tss:
                break
            ts_outfile_lst.append((ts, outfile))
        if ts_outfile_lst:
            _lst.append((tss, infile, ts_outfile_lst))
    kwargs["tss__infile__ts_outfile_lst__lst"] = _lst

    # -- Lonlat
    conf = kwargs["conf_lonlat"]

    conf["name_lon"], conf["name_lat"] = conf.pop("varnames")

    return kwargs


def cli():
    parser = setup_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    kwargs = vars(parser.parse_args())
    print_args(kwargs, skip=["tss"])
    preproc_args(parser, kwargs)
    main(**kwargs)


if __name__ == "__main__":
    cli()
