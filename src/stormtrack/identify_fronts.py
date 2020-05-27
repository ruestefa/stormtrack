#!/usr/bin/env python3

# Standard library
import argparse
import functools
import os
import sys
from copy import copy
from datetime import datetime
from multiprocessing import Pool
from multiprocessing import Value

# Third-party
import netCDF4 as nc4
import numpy as np

# Local
from .extra.io_misc import netcdf_derive_file
from .extra.fronts.fronts import identify_fronts
from .utils.various import extract_args
from .utils.various import print_args
from .utils.various import TimestepGenerator
from .utils.various import TimestepStringFormatter


its = Value("i", 0)


def main(conf_in, conf_out, conf_comp, conf_exe):

    conf_comp["verbose"] = conf_exe["vb"] > 1

    infiles_tss = conf_in["infiles_tss"]
    outfiles_tss = conf_out["outfiles_tss"]
    args_lst = [
        (ts, fi, outfiles_tss[(ts,)]) for (ts,), fi in sorted(infiles_tss.items())
    ]
    conf_exe["nts"] = len(infiles_tss)

    if not conf_exe["parallelize"]:
        for args in args_lst:
            identify_fronts_ts(conf_in, conf_comp, conf_out, conf_exe, *args)
    else:
        print("parallelize {}x".format(conf_exe["num_procs"]))
        fct = functools.partial(
            identify_fronts_ts, conf_in, conf_comp, conf_out, conf_exe
        )
        pool = Pool(
            conf_exe["num_procs"],
            maxtasksperchild=1,
            initializer=init_worker,
            initargs=(its,),
        )
        pool.starmap(fct, args_lst)

    print("100% [{}]".format(conf_in["timesteps"][-1]))


def init_worker(args):
    global its
    its = args


def identify_fronts_ts(conf_in, conf_comp, conf_out, conf_exe, ts, ifile, ofile):

    global its
    pp_ts = " {: 2.0%} [{}]".format(its.value / conf_exe["nts"], ts)

    if conf_exe["vb"] == 0:
        try:
            w, h = os.get_terminal_size()
        except OSError:
            pass
        else:
            print(pp_ts[:w], end="\r", flush=True)

    level = conf_in["level_hPa"]

    # Read input fields
    if conf_exe["vb"] >= 1:
        print(pp_ts + " read {}".format(ifile))
    iflds = read_fields(ifile, level, conf_comp["var_sfx"])

    # Identify fronts
    if conf_exe["vb"] >= 1:
        print(pp_ts + " identify frontal areas")
    kwas = copy(iflds)
    kwas["print_prefix"] = pp_ts
    kwas.update(conf_comp)
    oflds = identify_fronts(**kwas)
    for fld in oflds.values():
        fld[fld == conf_comp["nan"]] = np.nan

    # Write front fields to netCDF file
    if conf_exe["vb"] >= 1:
        print(pp_ts + " write {}".format(ofile))
    write_front_fields(
        ifile, ofile, oflds, level, conf_comp, conf_out, conf_comp["var_sfx"]
    )

    with its.get_lock():
        its.value += 1


# ==============================================================================


def read_fields(ifile, level, var_sfx=""):

    varnames = ["rlon", "rlat", "pressure", "T", "QV", "U", "V"]
    try:
        with nc4.Dataset(ifile, "r") as fi:
            try:
                rlon = fi.variables["rlon"][:]
                rlat = fi.variables["rlat"][:]
                P = fi.variables["pressure"][:]
                T = fi.variables["T" + var_sfx][:][0]
                QV = fi.variables["QV" + var_sfx][:][0]
                U = fi.variables["U" + var_sfx][:][0]
                V = fi.variables["V" + var_sfx][:][0]
                assert fi.variables["T" + var_sfx].dimensions[0] == "time"
            except KeyError as e:
                err = ("variable {} not found in file '{}'; available: {}").format(
                    e, ifile, ", ".join(fi.variables)
                )
                raise IOError(err) from None
    except OSError as e:
        if "No such file or directory" in str(e):
            raise FileNotFoundError(ifile)
        raise

    P /= 100  # Pa -> hPa

    n_levels = P.size

    if n_levels == 1:
        if P != level:
            err = "level {} hPa not in {}".format(level, P)
            raise Exception(err)
    else:
        # Get level index
        try:
            level_ind = P.tolist().index(level)
        except ValueError:
            err = "level {} hPa not in {}".format(level, P)
            raise Exception(err)

        # Extract level of interest
        T = T[level_ind]
        QV = QV[level_ind]
        U = U[level_ind]
        V = V[level_ind]

    T -= 273.15  # K -> degC

    destagger(U, axis=0)
    destagger(V, axis=1)

    # Create uniform pressure field
    P = np.ones(T.shape, T.dtype, order="F") * level

    return dict(lon=rlon, lat=rlat, P=P, T=T, QV=QV, U=U, V=V)


def destagger(fld, axis):
    """De-stagger 2d field in one direction by simple averaging."""
    if axis == 0:
        fld[:-1, :] = (fld[:-1, :] + fld[1:, :]) / 2
    elif axis == 1:
        fld[:, :-1] = (fld[:, :-1] + fld[:, 1:]) / 2
    else:
        raise NotImplementedError("axis {}".format(axis))


def write_front_fields(ifile, ofile, oflds, level, conf_comp, conf_out, var_sfx=""):

    nx, ny = oflds["fmask"].shape
    dims = [("time", 1), ("rlon", nx), ("rlat", ny)]

    title = "fronts on {} hPa".format(level)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    gattrs = {"title": title, "creation_date": timestamp}
    for key in ["tvar", "n_diffuse", "min_grad", "minsize"]:
        gattrs[key] = conf_comp[key]

    # -- Collect output fields

    tvar = conf_comp["tvar"]
    output = {}

    outvars = conf_out["outvars"]

    # Write fields to netCDF file
    with nc4.Dataset(ofile, "w") as fo:
        with nc4.Dataset(ifile, "r") as fi:
            # Transfer dimensions, lon/lat, etc.
            netcdf_derive_file(fi, fo)

            dims2d = ("time", "rlat", "rlon")
            ncatts_base = {
                "coordinates": "lon lat",
                "grid_mapping": "rotated_pole",
                "standard_name": "n/a",
                "long_name": "n/a",
                "units": "n/a",
            }

            if "finp" in outvars:
                fld = np.expand_dims(oflds["tfld"], axis=0)  # add time dim
                vo = fo.createVariable("finp_" + tvar, fld.dtype, dims2d)
                ncatts = ncatts_base  # SR_TMP
                vo.setncatts(ncatts)
                vo[:] = fld

            if "fmask" in outvars:
                fld = np.expand_dims(oflds["fmask"], axis=0)  # add time dim
                fld = fld.astype(np.uint8)
                vo = fo.createVariable("fmask_" + tvar, fld.dtype, dims2d)
                ncatts = ncatts_base  # SR_TMP
                vo.setncatts(ncatts)
                vo[:] = fld

            if "farea" in outvars:
                fld = np.expand_dims(oflds["farea"], axis=0)  # add time dim
                vo = fo.createVariable("farea_" + tvar, fld.dtype, dims2d)
                ncatts = ncatts_base  # SR_TMP
                vo.setncatts(ncatts)
                vo[:] = fld

            if "fvel" in outvars:
                fld = np.expand_dims(oflds["fvel"], axis=0)  # add time dim
                vo = fo.createVariable("fvel_" + tvar, fld.dtype, dims2d)
                ncatts = ncatts_base  # SR_TMP
                vo.setncatts(ncatts)
                vo[:] = fld


def setup_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input = parser_add_group__in(parser)
    output = parser_add_group__out(parser)
    comp = parser_add_group__comp(parser)
    exe = parser_add_group__exe(parser)

    return parser


def parser_add_group__in(parser):
    group = parser.add_argument_group("input")
    group.add_argument(
        "-i",
        "--infile-fmt",
        help=(
            "input file format string (with {YYYY} etc.); "
            "must have hourly frequency, i.e., contain all of "
            "{YYYY}, {MM}, {DD}, and {HH} at least once"
        ),
        metavar="fmt-str",
        dest="in__infile_fmt",
        required=True,
    )
    group.add_argument(
        "-s",
        "--timesteps",
        help="timesteps used to reconstruce input file names",
        metavar="YYYYMMDDHH",
        nargs="+",
        type=int,
        dest="in__tss",
    )
    group.add_argument(
        "-S",
        "--timesteps-range",
        help="timestep range (format: YYYYMMDDHH; stride in hours)",
        metavar=("start", "end", "stride"),
        nargs=3,
        action="append",
        type=int,
        dest="in__tss_ranges",
    )
    group.add_argument(
        "-l",
        "--level",
        help="level in hPa (e.g., 700 or 850)",
        type=int,
        metavar="level",
        default=850,
        dest="in__level_hPa",
    )
    return group


def parser_add_group__out(parser):
    group = parser.add_argument_group("output")
    group.add_argument(
        "-o",
        "--outfile-fmt",
        help=(
            "output file format string (with {YYYY} etc.); must "
            "same frequency as input files (i.e., hourly)"
        ),
        metavar="fmt-str",
        dest="out__outfile_fmt",
        required=True,
    )
    group.add_argument(
        "--outvars",
        help="fields written to disk",
        metavar="varname",
        choices=("finp", "fmask", "farea", "fvel"),
        nargs="+",
        default=["farea", "fvel"],
        dest="out__outvars",
    )
    group.add_argument(
        "--output-mode",
        help="how the fields are written to disk",
        metavar="mode",
        choices=("field", "list"),
        default="field",
        dest="out__mode",
    )
    return group


def parser_add_group__comp(parser, title="compute", preflag="", name="comp"):
    group = parser.add_argument_group(title)
    if preflag:
        preflag += "-"
    group.add_argument(
        "--{}temp-var".format(preflag),
        help="temperature variable",
        metavar="varname",
        choices=("T", "TH", "THE", "QV"),
        default="THE",
        dest=name + "__tvar",
    )
    group.add_argument(
        "--{}var-suffix".format(preflag),
        help="suffix of raw input variables",
        metavar="suffix",
        default="",
        dest=name + "__var_sfx",
    )
    group.add_argument(
        "--{}stride".format(preflag),
        help="stride in horizontal gradient computation",
        metavar="N",
        type=int,
        default=1,
        dest=name + "__stride",
    )
    group.add_argument(
        "--{}diffuse".format(preflag),
        help="apply diffusive smoothing filter N times",
        metavar="N",
        type=int,
        default=160,
        dest=name + "__n_diffuse",
    )
    group.add_argument(
        "--{}min-grad".format(preflag),
        help="minimum gradient to identify frontal areas",
        metavar="float",
        type=float,
        default=0,
        dest=name + "__min_grad",
    )
    group.add_argument(
        "--{}minsize".format(preflag),
        help="minimum cluster size in grid points for frontal areas",
        metavar="N",
        type=int,
        default=0,
        dest=name + "__minsize",
    )
    group.add_argument(
        "--{}nan".format(preflag),
        help="internally used missing value",
        type=float,
        metavar="float",
        default=-999,
        dest=name + "__nan",
    )
    return group


def parser_add_group__exe(parser):
    group = parser.add_argument_group("execution")
    group.add_argument(
        "-v",
        "--verbose",
        help="increase verbosity",
        action="count",
        default=0,
        dest="exe__vb",
    )
    group.add_argument(
        "--par",
        "--parallel",
        help="run parallelized",
        action="store_true",
        default=False,
        dest="exe__parallelize",
    )
    group.add_argument(
        "--seq",
        "--sequential",
        help="run sequentially",
        action="store_false",
        default=False,
        dest="exe__parallelize",
    )
    group.add_argument(
        "--num-procs",
        help="number of processes used for parallel execution",
        type=int,
        metavar="N",
        default=8,
        dest="exe__num_procs",
    )
    group.add_argument(
        "--profile",
        help="activate profiling",
        action="store_true",
        dest="exe__profile",
    )
    group.add_argument(
        "--profile-nlines",
        help="lines of profiling output shown",
        metavar="n",
        type=int,
        default=10,
        dest="exe__profile_nlines",
    )
    return group


def preproc_args(parser, kwas):
    for key in ["in", "out", "comp", "exe"]:
        kwas["conf_" + key] = extract_args(kwas, key)
    preproc_args__in(parser, kwas, kwas["conf_in"])
    preproc_args__out(parser, kwas, kwas["conf_out"])
    preproc_args__comp(parser, kwas, kwas["conf_comp"])
    preproc_args__exe(parser, kwas, kwas["conf_exe"])
    return kwas


def preproc_args__in(parser, kwas, conf):
    # Timesteps
    try:
        conf["timesteps"] = TimestepGenerator.from_args(
            conf.pop("tss"), conf.pop("tss_ranges")
        )
    except ValueError as e:
        parser.error(e)

    # Input files
    infile_fmt = conf.pop("infile_fmt")
    try:
        conf["infiles_tss"] = TimestepStringFormatter(infile_fmt, freq="hourly").run(
            conf["timesteps"]
        )
    except ValueError as e:
        parser.error(e)


def preproc_args__out(parser, kwas, conf):

    # Output files
    outfile_fmt = conf.pop("outfile_fmt")
    try:
        conf["outfiles_tss"] = TimestepStringFormatter(outfile_fmt, freq="hourly").run(
            kwas["conf_in"]["timesteps"]
        )
    except ValueError as e:
        parser.error(e)

    # SR_TMP<
    if conf["mode"] == "list":
        raise NotImplementedError("output mode: list")
    # SR_TMP>


def preproc_args__comp(parser, kwas, conf):
    conf["tvar"] = conf["tvar"].upper()


def preproc_args__exe(parser, kwas, conf):
    # SR_TMP<
    if conf["profile"]:
        raise NotImplementedError("profiling")
    # SR_TMP>


if __name__ == "__main__":
    parser = setup_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    args = parser.parse_args()
    try:
        print_args(args)
    except NameError:
        pass
    kwas = vars(args)
    preproc_args(parser, kwas)
    main(**kwas)
