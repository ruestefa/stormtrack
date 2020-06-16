#!/usr/bin/env python3

# Standard library
import sys
import os
import argparse
import functools
import logging as log
import pstats
import cProfile
from copy import copy
from datetime import datetime
from multiprocessing import Pool
from timeit import default_timer as timer
from pprint import pprint, pformat
from warnings import warn

# Thirt-party
import h5py
import netCDF4 as nc4
import numpy as np
import scipy as sp

# Local
from .core.constants import default_constants
from .core.identification import cyclones_to_features
from .core.identification import DEFAULT_TYPE_CODES
from .core.identification import features_grow
from .core.identification import identify_features as identify_features_core
from .core.io import write_feature_file
from .extra.cyclone_id.identify import identify_features as identify_cyclones_core
from .extra.cyclone_id import config as cycl_cfg
from .extra.fronts.fronts import identify_fronts
from .extra.io_misc import plot_cyclones_depressions_extrema
from .extra.utilities_misc import Field2D
from .extra.utilities_misc import threshold_at_timestep
from .identify_fronts import read_fields as fronts_read_raw_fields
from .utils.array import reduce_grid_resolution
from .utils.netcdf import nc_read_var_list
from .utils.netcdf import point_list_to_field
from .utils.various import extract_args
from .utils.various import ipython
from .utils.various import print_args
from .utils.various import TimestepGenerator
from .utils.various import TimestepStringFormatter


log.basicConfig(format="%(message)s", level=log.INFO)


def main(
    *, conf_in, conf_preproc, conf_out, conf_topo, conf_idfy, conf_split, conf_exe
):

    if conf_exe["profiling"]:
        pr = cProfile.Profile()
    else:
        pr = None

    timings = {}
    conf_exe["timings"] = timings
    if conf_exe["timings_measure"]:
        timings["tot"] = 0.0
        timings["tot_start"] = timer()
        timings["input_lst"] = []
        timings["input_start"] = timer()
        timings["core_lst"] = []

    # Read grid parameters
    lon, lat = read_lonlat(conf_in)
    nx, ny = lon.shape
    conf_in["nx"] = nx
    conf_in["ny"] = ny

    # Read topography (if necessary)
    read_topo(conf_topo, conf_in)

    if conf_exe["timings_measure"]:
        timings["input_lst"].append(timer() - timings.pop("input_start"))

    # Relate input and output files to each other
    infiles_per_outfile = relate_files_ts(
        conf_out["outfiles_ts"], conf_in["infiles_tss"]
    )

    # For each output file, read all necessary fields,
    # identify all features, and write them to disk
    if conf_exe["parallelize"]:
        read_identify_write_par(
            conf_in,
            conf_out,
            conf_topo,
            conf_preproc,
            conf_idfy,
            conf_split,
            conf_exe,
            lon,
            lat,
            infiles_per_outfile,
        )
    else:
        read_identify_write_seq(
            conf_in,
            conf_out,
            conf_topo,
            conf_preproc,
            conf_idfy,
            conf_split,
            conf_exe,
            lon,
            lat,
            infiles_per_outfile,
        )

    if conf_exe["timings_measure"]:
        timings["tot"] = timer() - timings.pop("tot_start")
        for key in timings.copy():
            if key.endswith("_lst"):
                timings[key[:-4]] = np.sum(timings.pop(key))
        print("\n timings:")
        for key, val in sorted(timings.items()):
            print(" {:20} : {:10.3f}s".format(key, val))
        print("")
        timings_logfile = conf_exe["timings_logfile"]
        if timings_logfile:
            print("write timings to {}".format(timings_logfile))
            with open(timings_logfile, "w") as fo:
                for key, val in sorted(timings.items()):
                    fo.write(" {:20} {:10.3f}\n".format(key, val))
            print("")

    if conf_exe["profiling"]:
        ps = pstats.Stats(pr)
        nlines = 20
        ps.strip_dirs().sort_stats("tottime").print_stats(nlines)


def read_lonlat(conf):

    infile = conf["infile_lonlat"]
    name_lon, name_lat = conf["lonlat_names"]
    transpose = conf["lonlat_transp"]

    # Read fields from file
    if infile.endswith(".npz"):
        with np.load(infile) as fi:
            lon, lat = fi[name_lon], fi[name_lat]
    elif infile.endswith("h5"):
        with h5py.File(infile) as fi:
            lon, lat = fi[name_lon], fi[name_lat]
    else:
        print("< read " + infile)  # SR_DBG
        with nc4.Dataset(infile, "r") as fi:
            lon = fi["lon"][:]
            lat = fi["lat"][:]

    if transpose:
        lon = lon.T
        lat = lat.T

    if conf.get("reduce_grid_resolution", False):
        # Reduce the grid resolution by striding
        lon = reduce_grid_resolution(lon, conf["reduce_grid_stride"], "pick")
        lat = reduce_grid_resolution(lat, conf["reduce_grid_stride"], "pick")

    return lon, lat


def read_topo(conf_topo, conf_in):

    varname = conf_topo["varname"]
    filter_threshold = conf_topo["filter_threshold"]
    infile = conf_topo["infile"]

    reduce_stride = conf_in["reduce_grid_stride"]
    reduce_mode = conf_in["reduce_grid_mode"]

    if filter_threshold < 0:
        conf_topo["fld"] = None
        conf_topo["filter_apply"] = False
        return

    log.info("read topography field {} from {}".format(varname, infile))
    if infile.endswith(".npz"):
        # Numpy archive
        with np.load(infile) as fi:
            topo_fld = fi[varname]
    else:
        # Assuming netcdf
        with nc4.Dataset(infile, "r") as fi:
            topo_fld = fi[varname][0]

    conf_topo["fld"] = topo_fld
    conf_topo["filter_apply"] = True

    if reduce_stride > 1:
        conf_topo["fld"] = reduce_grid_resolution(
            conf_topo["fld"], reduce_stride, reduce_mode
        )


def read_identify_write_seq(
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_exe,
    lon,
    lat,
    infiles_per_outfile,
):

    for timesteps_out__outfile in sorted(conf_out["outfiles_ts"].items()):
        timings_i = read_identify_write_core(
            conf_in,
            conf_out,
            conf_topo,
            conf_preproc,
            conf_idfy,
            conf_split,
            conf_exe,
            lon,
            lat,
            infiles_per_outfile,
            conf_exe["timings"],
            timesteps_out__outfile,
        )


def read_identify_write_par(
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_exe,
    lon,
    lat,
    infiles_per_outfile,
):

    # Prepare clean timings dict passed to parallel processes
    timings_clean = {}
    for key, val in conf_exe["timings"].items():
        if key.endswith("_start"):
            continue
        elif isinstance(val, list):
            timings_clean[key] = []
        else:
            timings_clean[key] = 0.0

    # Sort outfiles into chunks and launches for parallelization
    num_procs = conf_exe["num_procs"]
    max_per_proc = 23 * 7
    outfiles_ts_lst = sorted(conf_out["outfiles_ts"].items())
    n_outfiles_tot = len(outfiles_ts_lst)
    n_launches = int(float(n_outfiles_tot) / (num_procs * max_per_proc) + 1)
    n_per_launch = int(float(n_outfiles_tot) / n_launches + 1)
    n_chunks = num_procs
    chunks_launches = [[[] for _ in range(n_chunks)] for _ in range(n_launches)]
    i_launch, i_chunk, n_launch_i = 0, 0, 0
    for outfile_ts in sorted(conf_out["outfiles_ts"].items()):
        chunks_launches[i_launch][i_chunk].append(outfile_ts)
        n_launch_i += 1
        if n_launch_i == n_per_launch:
            i_launch += 1
            i_chunk = 0
            continue
        i_chunk = (i_chunk + 1) % n_chunks

    print("\n" + ("+" * 40))
    print(
        (
            "parallelize {} outfiles over {} launches with {} threads " "(max. {} each)"
        ).format(n_outfiles_tot, n_launches, num_procs, max_per_proc)
    )
    print(("+" * 40) + "\n")

    # Run in parallel
    fct = functools.partial(
        read_identify_write_par_core,
        conf_in,
        conf_out,
        conf_topo,
        conf_preproc,
        conf_idfy,
        conf_split,
        conf_exe,
        lon,
        lat,
        infiles_per_outfile,
        timings_clean,
    )
    pool = Pool(num_procs, maxtasksperchild=1)
    results = []
    for chunks_launch in chunks_launches:
        results_launch = pool.map(fct, chunks_launch)
        results.extend(results_launch)

    # Merge timings from parallel regions into original timings dict
    for timings_i in results:
        for key, val in timings_i.items():
            if not key.endswith("_start"):
                if key in conf_exe["timings"]:
                    conf_exe["timings"][key] += val
                else:
                    conf_exe["timings"][key] = val


def read_identify_write_par_core(
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_exe,
    lon,
    lat,
    infiles_per_outfile,
    timings,
    outfiles_ts,
):
    for timesteps_out__outfile in outfiles_ts:
        timings_i = read_identify_write_core(
            conf_in,
            conf_out,
            conf_topo,
            conf_preproc,
            conf_idfy,
            conf_split,
            conf_exe,
            lon,
            lat,
            infiles_per_outfile,
            timings,
            timesteps_out__outfile,
        )
    return timings


def read_identify_write_core(
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_exe,
    lon,
    lat,
    infiles_per_outfile,
    timings,
    timesteps_out__outfile,
):
    """Read input fields, identify features, and write them to disk."""
    timesteps, outfile = timesteps_out__outfile

    infiles = infiles_per_outfile[outfile]
    infiles_tss_i = {
        tss_in: infile
        for tss_in, infile in conf_in["infiles_tss"].items()
        if infile in infiles
    }

    # Identify features
    feature_name = conf_out["feature_name"]
    if conf_exe["profiling"] and pr:
        pr.enable()
    features = []
    nts = len(timesteps)
    for its, timestep in enumerate(timesteps):
        new_features, timings_i = identify_features(
            name=conf_out["feature_name"],
            conf_in=conf_in,
            conf_topo=conf_topo,
            conf_preproc=conf_preproc,
            conf_idfy=conf_idfy,
            conf_split=conf_split,
            lon=lon,
            lat=lat,
            timestep=timestep,
            its=its,
            nts=nts,
            timings_measure=conf_exe["timings_measure"],
        )
        features.extend(new_features)
    if conf_exe["profiling"] and pr:
        pr.disable()

    # Store info and features
    # SR_TODO remove lower/upper/... from arguments
    # SR_TODO maybe change such that the whole "info" dict is passed in
    info_features = get_info_features(conf_in, conf_idfy, conf_split, conf_topo)

    if conf_exe["timings_measure"]:
        timings["output_start"] = timer()

    # Write features to disk
    write_feature_file(
        outfile,
        feature_name=feature_name,
        features=features,
        info_features=info_features,
        nx=conf_in["nx"],
        ny=conf_in["ny"],
    )

    if conf_exe["timings_measure"]:
        timings["output"] = timer() - timings.pop("output_start")

    return timings


def get_info_features(conf_in, conf_idfy, conf_split, conf_topo):
    return dict(
        varname=conf_in["varname"],
        thresholds=conf_idfy["thresholds"],
        minsize=conf_idfy["minsize"],
        split_levels=conf_split["levels"],
        split_seed_minsize=conf_split["seed_minsize"],
        split_seed_minstrength=conf_split["seed_minstrength"],
        topo_filter_apply=conf_topo["filter_apply"],
        topo_filter_mode=conf_topo["filter_mode"],
        topo_filter_threshold=conf_topo["filter_threshold"],
        topo_filter_min_overlap=conf_topo["filter_min_overlap"],
    )


def relate_files_ts(files1_ts, files2_ts):
    """Relate two sets of timestep-sorted file names with each other."""
    files2_file1 = {}
    for tss1, file1 in files1_ts.items():
        files2_file1[file1] = [
            file2
            for tss2, file2 in files2_ts.items()
            if any(ts1 in tss2 for ts1 in tss1)
        ]
    return files2_file1


def identify_features(
    *,
    name,
    conf_in,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    lon,
    lat,
    timestep,
    timings_measure,
    grid=None,
    silent=False,
    its=None,
    nts=None,
):
    timings = {}

    if timings_measure:
        timings["input_start"] = timer()

    # Get input file name
    infile = get_infile(conf_in, timestep)
    if not silent:
        if its is None:
            its = -2
        if nts is None:
            nts = -1
        log.info("input file {}/{}: {}".format(its + 1, nts, infile))

    # Set the base id: timestep (10) + type code (3) + id (4)
    base_id = (timestep * 1e3 + conf_idfy["type_code"]) * 1e4

    if conf_in["comp_special"] in ["cyclones", "anticyclones"]:

        # SR_TMP<
        if conf_in["reduce_grid_resolution"]:
            raise NotImplementedError("cyclones with reduced grid resolution")
        # SR_TMP>

        # Special case: identify cyclone features
        flds_named, new_features = identify_cyclones(
            infile,
            name,
            conf_in,
            conf_preproc,
            timestep,
            anti=conf_in["comp_special"].startswith("anti"),
        )

        if timings_measure:
            timings["input_lst"] = [timer() - timings.pop("input_start")]
            timings["core_start"] = timer()

    else:
        # -- General case: identify features

        # SR_TMP<
        if conf_in.get("crop_domain_n", 0) > 0:
            raise NotImplementedError("crop domain")
        # SR_TMP>

        # Read fields used for identification
        if conf_in["comp_special"] == "fronts":
            # Special case: compute front fields from raw fields
            flds_named = compute_front_fields(
                infile, conf_in, conf_preproc, conf_in["conf_comp_fronts"]
            )
        else:
            # General case: read raw fields
            flds_named = import_fields(infile, lon, lat, conf_in, conf_preproc)

        if timings_measure:
            timings["input_lst"] = [timer() - timings.pop("input_start")]
            timings["core_start"] = timer()

        # Identify the features
        # SR_TODO clean up thresholds: currently, -1 is used internally
        # SR_TODO for omitted threshold, thus limiting the identification
        # SR_TODO to positive fields, which, e.g., vertical wind is not
        # SR_TODO Use some "missing value" as default (e.g. the smallest/
        # SR_TODO biggest number for the given float type)!
        lower = threshold_at_timestep(conf_idfy["thresholds"][0], timestep)
        upper = threshold_at_timestep(conf_idfy["thresholds"][1], timestep)
        new_features = identify_features_core(
            fld=flds_named[conf_in["varname"]],
            feature_name=name,
            nx=conf_in["nx"],
            ny=conf_in["ny"],
            base_id=base_id,
            timestep=timestep,
            lower=lower,
            upper=upper,
            minsize=conf_idfy["minsize"],
            maxsize=conf_idfy["maxsize"],
            split_levels=conf_split["levels"],
            split_seed_minsize=conf_split["seed_minsize"],
            split_seed_minstrength=conf_split["seed_minstrength"],
            topo_filter_apply=conf_topo["filter_apply"],
            topo_filter_mode=conf_topo["filter_mode"],
            topo_filter_threshold=conf_topo["filter_threshold"],
            topo_filter_min_overlap=conf_topo["filter_min_overlap"],
            topo_fld=conf_topo["fld"],
            silent=silent,
            grid=grid,
        )

        if timings_measure:
            timings["core_lst"] = [timer() - timings.pop("core_start")]

    if not silent:
        log.info("identified {} '{}' features".format(len(new_features), name))

    if timings_measure:
        timings["core_start"] = timer()

    postproc_features(new_features, flds_named, infile, lon, lat, conf_in)

    if conf_idfy["grow_features_n"]:
        const = default_constants(nx=conf_in["nx"], ny=conf_in["ny"])
        features_grow(
            conf_idfy["grow_features_n"],
            new_features,
            const,
            inplace=True,
            retain_orig=True,
        )

    if timings_measure:
        timings["core_lst"] = [timer() - timings.pop("core_start")]

    return new_features, timings


def get_infile(conf_in, timestep):
    for timesteps in conf_in["infiles_tss"]:
        if timestep in timesteps:
            infile = conf_in["infiles_tss"][timesteps]
            break
    else:
        err = "no infile found for timestep {}:\n{}".format(
            timestep, pformat(conf_in["infiles_tss"])
        )
        raise Exception(err)
    return infile


def import_fields(infile, lon, lat, conf_in, conf_preproc):
    """Read a field from disk (netCDF, npz) and preprocess it."""

    # Fetch some arguments
    informat = conf_in["input_format"]
    fld_name = conf_in["varname"]
    transpose = conf_in["infield_transpose"]
    fld_mirror = conf_in["infield_mirror"]
    level = conf_in["infield_lvl"]
    reduce_stride = conf_in["reduce_grid_stride"]
    reduce_mode = conf_in["reduce_grid_mode"]

    # Read field (or compute raw fields in special cases)
    fld = read_field(
        infile,
        fld_name,
        level,
        transpose,
        fld_mirror,
        informat,
        reduce_stride,
        reduce_mode,
        lon,
        lat,
    )
    flds_named = {fld_name: fld}

    # Masking
    mask_name = conf_preproc["mask_varname"]
    if mask_name:
        fld_mask = read_field(
            infile,
            mask_name,
            level,
            transpose,
            conf_preproc["mask_mirror"],
            informat,
            reduce_stride,
            reduce_mode,
            lon,
            lat,
        )
        mask_field(
            fld,
            fld_mask,
            conf_preproc["mask_threshold_gt"],
            conf_preproc["mask_threshold_lt"],
        )

    # Boundary filter
    trim_boundaries(fld, conf_preproc["trim_boundaries_n"])

    # Add refval
    if conf_preproc["add_refval"]:
        fld += conf_preproc["refval"]

    replace_varname = conf_in["replace_varname"]
    if replace_varname:
        # Read field used to replace feature values in the end
        informat = conf_in["input_format"]
        fld_name = conf_in["replace_varname"]
        level = conf_in["infield_lvl"]
        transpose = conf_in["infield_transpose"]
        mirror = False
        fld_repl = read_field(
            infile,
            replace_varname,
            level,
            transpose,
            mirror,
            informat,
            reduce_stride,
            reduce_mode,
            lon,
            lat,
        )
        flds_named[replace_varname] = fld_repl

    return flds_named


def compute_front_fields(infile, conf_in, conf_preproc, conf_comp_fronts):
    """Read a field from disk (netCDF, npz) and preprocess it."""

    # Fetch some input arguments
    informat = conf_in["input_format"]
    fld_name = conf_in["varname"]
    transpose = conf_in["infield_transpose"]
    fld_mirror = conf_in["infield_mirror"]
    level = conf_in["infield_lvl"]
    reduce_stride = conf_in["reduce_grid_stride"]
    reduce_mode = conf_in["reduce_grid_mode"]

    # Fetch some preprocessing arguments
    mask_name = conf_preproc["mask_varname"]
    mask_mirror = conf_preproc["mask_mirror"]
    mask_threshold_gt = conf_preproc["mask_threshold_gt"]
    mask_threshold_lt = conf_preproc["mask_threshold_lt"]
    trim_boundaries_n = conf_preproc["trim_boundaries_n"]
    refval = conf_preproc["refval"]
    add_refval = conf_preproc["add_refval"]

    if informat != "field":
        err = ("input format must be 'field' to compute fronts, not '{}'").format(
            informat
        )
        raise ValueError(err)

    # Read raw input fields
    iflds = fronts_read_raw_fields(infile, level, var_sfx=conf_comp_fronts["var_sfx"])

    if reduce_stride > 1:
        # Reduce grid resolution by striding
        for name, fld in iflds.items():
            iflds[name] = reduce_grid_resolution(fld, reduce_stride, reduce_mode)

    # Compute front fields
    kwas = copy(iflds)
    kwas.update(conf_comp_fronts)
    # kwas["print_prefix"] = ""
    # kwas["verbose"] = False
    oflds = identify_fronts(**kwas)
    _fmt = lambda s: "{}_{}{}".format(
        s, conf_comp_fronts["tvar"], ""  # conf_comp_fronts["var_sfx"]
    )
    oflds = {_fmt(k): v for k, v in oflds.items()}
    fld = oflds[fld_name]

    # Masking
    if mask_name:
        fld_mask = oflds[mask_name]
        if mask_mirror:
            fld_mask = -fld_mask
        mask_field(fld, fld_mask, mask_threshold_gt, mask_threshold_lt)

    # Add reference value
    if add_refval:
        fld += refval

    # Boundary filter
    trim_boundaries(fld, trim_boundaries_n)

    return oflds


def read_field(
    infile,
    varname,
    level,
    transpose,
    mirror,
    input_format="field",
    reduce_stride=1,
    reduce_mode="mean",
    lon=None,
    lat=None,
):

    # Read field
    if "field" in input_format:
        fld = read_field__field(infile, varname, level)

        if reduce_stride > 1:
            fld = reduce_grid_resolution(fld, reduce_stride, reduce_mode)

    elif "list" in input_format:
        if lon is None or lat is None:
            err = "must pass lon, lat to read list input"
            raise ValueError(err)

        # SR_TMP<
        if reduce_stride > 1:
            # Note: original lon, lat not available
            raise NotImplementedError("reduce stride for list input")
        # SR_TMP>

        fld = read_field__list(infile, varname, lon, lat)

    else:
        err = "invalid input format '{}'".format(input_format)
        raise ValueError(err)

    # Transpose if necessary
    if transpose:
        fld = fld.T

    # Mirror if necessary
    if mirror:
        fld = -fld

    return fld


def read_field__field(infile, varname, level):
    """Read normal 2D field."""

    # Read field ...
    if infile.endswith(".npz"):
        # ...from numpy archive
        with np.load(infile) as fi:
            fld = fi[varname]
    elif infile.endswith(".h5"):
        # ...from hdf5 archive
        with h5py.File(infile, "r") as fi:
            fld = fi[varname][:]
    else:
        # ... from netcdf file
        try:
            with nc4.Dataset(infile, "r") as fi:
                vi = fi.variables[varname]
                fld = vi[:]
                if vi.dimensions[0] == "time":
                    fld = fld[0]
        except Exception as e:
            err = ("error reading field '{}' from file '{}':\n{}({})").format(
                varname, infile, e.__class__.__name__, e
            )
            raise Exception(err)
    if np.ma.is_masked(fld):
        fld = fld.filled()

    # Check dimensions
    if len(fld.shape) != 2:
        if len(fld.shape) == 3:
            if level is None:
                err = "must pass level for 3d input field: {}".format(varname)
                raise ValueError(err)
            fld = fld[level, :, :]
        else:
            err = "wrong number of dimensions: {} != 2 {}".format(
                len(fld.shape), fld.shape
            )
            raise Exception(err)

    return fld


def read_field__list(infile, varname, lon, lat):
    """Read point list and initialize field."""

    # Read field ...
    if infile.endswith(".npz"):
        # ... from numpy archive
        with np.load(infile) as fi:
            pts_lon, pts_lat, pts_fld = fi["lon"], fi["lat"], fi[varname]
        fld = point_list_to_field(pts_lon, pts_lat, pts_fld, lon, lat)
    elif infile.endswith(".h5"):
        # .. from hdf5 file
        with h5py.File(infile, "r") as fi:
            pts_lon, pts_lat, pts_fld = fi["lon"], fi["lat"], fi[varname]
        fld = point_list_to_field(pts_lon, pts_lat, pts_fld, lon, lat)
    else:
        # ... from netcdf file
        fld = nc_read_var_list(infile, varname, lon, lat)

    return fld


def mask_field(fld, fld_mask, lower, upper):
    if upper is None:
        mask = fld_mask > lower
    elif lower is None:
        mask = fld_mask < upper
    else:
        mask = (fld_mask > lower) & (fld_mask < mask_lt)
    fld[~mask] = 0


def trim_boundaries(fld, n):
    if n > 0:
        fld[:n, :] = 0
        fld[:, :n] = 0
        fld[-n:, :] = 0
        fld[:, -n:] = 0


def postproc_features(features, flds_named, infile, lon, lat, conf_in):

    # If field is mirrored, reverse values
    if conf_in["infield_mirror"]:
        for feature in features:
            feature.mirror_values()

    if conf_in["replace_varname"]:
        # Replace feature values
        fld = flds_named[conf_in["replace_varname"]]
        for feature in features:
            feature.replace_values(fld)

    if conf_in["minsize_km2"] and conf_in["minsize_km2"] > 1:
        for feature in [f for f in features]:
            if feature.area_lonlat(lon, lat, "km2") < conf_in["minsize_km2"]:
                features.remove(feature)

    if conf_in["maxsize_km2"] and conf_in["maxsize_km2"] > 0:
        for feature in [f for f in features]:
            if feature.area_lonlat(lon, lat, "km2") > conf_in["maxsize_km2"]:
                features.remove(feature)


def identify_cyclones(infile, name, conf_in, conf_preproc, timestep, anti=False):

    fld_name = conf_in["varname"]

    inifile = conf_in["cycl_inifile"]
    if inifile is None:
        raise Exception("must pass cyclones inifile")

    # SR_TMP< TODO use command line arguments
    topo = None
    fact = 0.01  # convert Pa to hPa or geopotential to gpdm
    # SR_TMP>

    # Set up config: merge inifile config into default config
    conf_def = cycl_cfg.get_config_default()
    conf_ini = cycl_cfg.get_config_inifile(inifile)
    conf = cycl_cfg.merge_configs([conf_def, conf_ini])
    conf["IDENTIFY"]["timings-identify"] = None
    conf["IDENTIFY"]["datetime"] = timestep

    # Fetch some config values
    level = conf_in["infield_lvl"]
    if level is None:
        level = conf["GENERAL"]["input-field-level"]
    fld_name = conf_in["varname"]
    if fld_name is None:
        fld_name = conf["GENERAL"]["input-field-name"]

    # Read pressure or height field
    fld = read_input_field_lonlat(
        infile,
        fld_name,
        level,
        "lon",
        "lat",
        conv_fact=fact,
        crop=conf_preproc["crop_domain_n"],
    )

    refval = conf_preproc["refval"]
    # if refval is not None:
    #    refval *= fact

    if conf_preproc["add_refval"]:
        fld += refval

    if anti:
        # Mirror field around reference value (e.g., 1015 hPa for SLP)
        if refval is None:
            err = (
                "must provide reference value to identify anticyclones "
                "('refval' in conf_preproc)"
            )
            raise Exception(err)
        _m = ~np.isnan(fld)
        fld[_m] = -1 * (fld[_m] - refval) + refval

    # SR_TMP<
    if conf_preproc["trim_boundaries_n"] > 0:
        err = "cyclones: --trim-boundaries; consider --shrink-domain"
        raise NotImplementedError(err)
    if conf_preproc["crop_domain_n"] > 0:
        pass
    # SR_TMP>

    # Identify cyclone features
    _r = identify_cyclones_core(fld, topo, conf["IDENTIFY"])
    cyclones = _r["cyclones"]
    depressions = _r["depressions"]

    # Add type code to cyclone features
    tcode = DEFAULT_TYPE_CODES[name]
    for cyclone in cyclones:
        assert str(cyclone._id).startswith(str(timestep))
        _core = str(cyclone._id)[len(str(timestep)) :]
        cyclone._id = int(str(timestep) + str(tcode) + _core)

    # Plot cyclones, depressions, and/or extrema
    if conf["GENERAL"]["make-plots"]:
        plot_cyclones_depressions_extrema(infile, cyclones, depressions, fld, conf)

    if anti:
        # Mirror field around reference value (e.g., 1015 hPa for SLP)
        _m = ~np.isnan(fld)
        fld[_m] = -1 * (fld[_m] - refval) + refval

    # Convert cyclone features to new-style features
    features = []
    cyclones_to_features(
        timestep, cyclones, fld, fld.lon, fld.lat, out=features, vb=False
    )

    flds_by_name = {fld_name: fld}
    return flds_by_name, features


def read_input_field_lonlat(
    input_file, fld_name, level, lon_name, lat_name, conv_fact=None, crop=0
):
    """Read from file and pre-process a field.

    Returns the field as a Field2D object.

    Arguments:
     - input_file: Input netCDF file.
     - fld_name: Name of the input field used in the input file.
     - lon_name: Name of the longitude field in the input file.
     - lat_name: Name of the latitude field in the input file.

    Optional arguments:
     - conv_fact: Conversion factor applied to the field.
     - crop: cut N pixels off around the domain
    """
    # Read the raw field from file, along with lon/lat fields
    try:
        with nc4.Dataset(input_file, "r") as fi:
            lon = fi[lon_name][:]
            lat = fi[lat_name][:]
            fld_raw = fi[fld_name][0]  # strip leading time dimension
    except Exception as e:
        err = "Cannot read '{}' from {}\n{}: {}".format(
            fld_name, input_file, e.__class__.__name__, str(e).strip()
        )
        raise IOError(err)

    # Shrink domain
    if crop is not None and crop > 0:
        fld_raw = fld_raw[crop:-crop, crop:-crop]
        lon = lon[crop:-crop, crop:-crop]
        lat = lat[crop:-crop, crop:-crop]

    # Select level
    if level is not None:
        fld_raw = fld_raw[level, :, :]

    # Apply a conversion factor
    if conv_fact is not None:
        fld_raw *= conv_fact

    # Create a Field2D object
    fld = Field2D(fld_raw, lon, lat)

    return fld


# SR_TMP< TODO solve issue with compoling fronts._libfronts on daint
try:
    from .identify_fronts import parser_add_group__comp as parser_add_group__comp_fronts
    from .identify_fronts import preproc_args__comp as preproc_args__comp_fronts
except ImportError as e:
    msg = ("warning: fronts-related import failed: {}; cannot identify fronts!").format(
        e
    )
    print(msg)

    def parser_add_group__comp_fronts(*args, **kwas):
        pass

    def preproc_args__comp_fronts(*args, **kwas):
        pass


# SR_TMP>


description = """Identify features based on thresholds etc."""


def setup_parser():
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input = parser_add_group__in__base(parser)
    input = parser_extend_group__in(parser, input)
    parser_add_group__out(input)
    parser_add_group__preproc(parser)
    parser_add_group__topo(parser)
    parser_add_group__idfy(parser)
    parser_add_group__split(parser)
    parser_add_group__exe(parser)

    return parser


def parser_add_group__in__base(parser):
    group = parser.add_argument_group("input")

    group.add_argument(
        "-i",
        "--infile-fmt",
        help="input file format (netcdf, with {YYYY} etc.)",
        metavar="str",
        dest="in__infile_format",
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
        dest="in__ts_ranges",
    )
    group.add_argument(
        "--infile-lonlat",
        help="file containing lon/lat",
        metavar="npz-file",
        default="crclim_lonlat.npz",
        dest="in__infile_lonlat",
    )
    group.add_argument(
        "--lonlat-names",
        help="names of lon and lat variables (lon/lat or rlon/rlat)",
        nargs=2,
        default=["rlon", "rlat"],
        dest="in__lonlat_names",
    )
    group.add_argument(
        "--lonlat-transpose",
        help="transpose lon/lat fields",
        action="store_true",
        dest="in__lonlat_transp",
    )

    return group


def parser_extend_group__in(parser, group):
    group.add_argument(
        "--fld-varname",
        help="name of variable used for identification",
        metavar="varname",
        dest="in__varname",
    )
    group.add_argument(
        "--lvl",
        "--level",
        help="level (for 3d input data)",
        type=int,
        metavar="lvl",
        dest="in__infield_lvl",
    )
    group.add_argument(
        "-f",
        "--input-format",
        help="format of data in netcdf file",
        metavar="format",
        choices=("field", "list"),
        default="field",
        dest="in__input_format",
    )
    group.add_argument(
        "--transpose",
        help="transpose input field (and mask field, if provided)",
        action="store_true",
        dest="in__infield_transpose",
    )
    group.add_argument(
        "--fld-mirror",
        help=(
            "mirror the input field values during the "
            "identification step; note that they are written "
            "to disk non-mirrored"
        ),
        action="store_true",
        default=False,
        dest="in__infield_mirror",
    )
    group.add_argument(
        "--fld-nomirror",
        help=("don't mirror the input field values during the " "identification step"),
        action="store_false",
        default=False,
        dest="in__infield_mirror",
    )
    group.add_argument(
        "--reduce-grid-stride",
        help="reduce grid resolution by striding the input field",
        metavar="N",
        type=int,
        choices=tuple(range(1, 100, 2)),
        default=1,
        dest="in__reduce_grid_stride",
    )
    group.add_argument(
        "--reduce-grid-mode",
        help=("how to reduce the values from the original to the " "strided grid"),
        metavar="mode",
        choices=("pick", "mean",),
        default="mean",
        dest="in__reduce_grid_mode",
    )
    group.add_argument(
        "--replace-varname",
        help=(
            "name of variable which replaces feature values after "
            "the identification step"
        ),
        metavar="varname",
        dest="in__replace_varname",
    )
    group.add_argument(
        "--comp",
        help=(
            "compute front fields from raw COSMO output fields "
            "(input fields: rlon, rlat, pressure, T, QV, U, V)"
        ),
        metavar="type",
        choices=("fronts", "cyclones", "anticyclones"),
        default=[],
        dest="in__comp_special",
    )
    # SR_TMP<
    group.add_argument(
        "--cyclones-inifile",
        help="mandatory inifile for cyclone identification",
        metavar="inifile",
        dest="in__cycl_inifile",
    )
    # SR_TMP>

    # SR_TODO find a cleaner solution
    # SR_TODO (because it's here, it's added automatically in track_features)
    parser_add_group__comp_fronts(
        parser, "compute fronts", preflag="fronts", name="in__comp_fronts"
    )

    return group


def parser_add_group__out(parser):
    group = parser.add_argument_group("output")

    group.add_argument(
        "--feature-name",
        help="output feature name",
        metavar="str",
        dest="out__feature_name",
        required=True,
    )
    group.add_argument(
        "-o",
        "--outfile-fmt",
        help="output file format (npz, with path and {YYYY} etc.)",
        metavar="str",
        dest="out__outfile_format",
        required=True,
    )

    return group


def parser_add_group__preproc(parser):
    group = parser.add_argument_group("pre-processing")

    group.add_argument(
        "-b",
        "--trim-boundaries",
        help="trim N pixels from all four boundaries",
        metavar="N",
        type=int,
        default=0,
        dest="preproc__trim_boundaries_n",
    )
    group.add_argument(
        "--crop-domain",
        help="cut off N pixels around the domain",
        metavar="N",
        type=int,
        default=0,
        dest="preproc__crop_domain_n",
    )
    group.add_argument(
        "--mv",
        "--mask-varname",
        help=(
            "name of variable used for masking; if passed, the "
            "input field is masked before the identification "
            "step, using this field with threshold(s)"
        ),
        metavar="varname",
        dest="preproc__mask_varname",
    )
    group.add_argument(
        "--mask-mirror",
        help="mirror the mask field values",
        action="store_true",
        default=False,
        dest="preproc__mask_mirror",
    )
    group.add_argument(
        "--mask-nomirror",
        help="don't mirror the mask field values",
        action="store_false",
        default=False,
        dest="preproc__mask_mirror",
    )
    group.add_argument(
        "--mask-gt",
        help=(
            "lower threshold for masking; pass lower and/or upper "
            "threshold for masking"
        ),
        type=float,
        metavar="threshold",
        dest="preproc__mask_threshold_gt",
    )
    group.add_argument(
        "--mask-lt",
        help=(
            "upper threshold for masking; pass lower and/or upper "
            "threshold for masking"
        ),
        type=float,
        metavar="threshold",
        dest="preproc__mask_threshold_lt",
    )
    group.add_argument(
        "--mask-and-or",
        help=(
            "whether both ('and') or only one ('or') threshold "
            "condition must be fulfilled"
        ),
        choices=("and", "or"),
        default="and",
        dest="preproc__mask_and_or",
    )
    group.add_argument(
        "--reference-value",
        help=(
            "reference value for input field, necessary for "
            "instance to identify anticyclones"
        ),
        type=float,
        metavar="val",
        dest="preproc__refval",
    )
    group.add_argument(
        "--add-reference-value",
        help=(
            "add reference value to fields, which can be useful "
            "if the input fields constitute anomalies"
        ),
        action="store_true",
        dest="preproc__add_refval",
    )

    return group


def parser_add_group__topo(parser):
    group = parser.add_argument_group("topography filter")

    group.add_argument(
        "--topo-file",
        help="file which contains topography",
        metavar="file",
        default="./crclim_const.nc",
        dest="topo__infile",
    )
    group.add_argument(
        "--topo-var",
        help="name of topography field",
        metavar="name",
        default="HSURF",
        dest="topo__varname",
    )
    group.add_argument(
        "--topo-filter-mode",
        help="when and how to apply the topography filter",
        choices=("features", "pixels"),
        default="pixels",
        dest="topo__filter_mode",
    )
    group.add_argument(
        "--topo-filter-threshold",
        help="threshold for topography filter",
        metavar="float",
        type=float,
        default=-1,
        dest="topo__filter_threshold",
    )
    group.add_argument(
        "--topo-filter-min-overlap",
        help="min. rel. overlap of feature with mask",
        metavar="fraction",
        type=float,
        default=-1,
        dest="topo__filter_min_overlap",
    )

    return group


def parser_add_group__idfy(parser):
    group = parser.add_argument_group("identification")

    group.add_argument(
        "-t",
        "--type-code",
        help=(
            "3-digit feature type codes for feature id; can be "
            "omitted for common feature types for which default "
            "are defined"
        ),
        metavar="iii",
        type=int,
        dest="idfy__type_code",
    )
    group.add_argument(
        "--id-format",
        help="feature id format (count or datetime)",
        choices=("count", "datetime"),
        metavar="str",
        default="datetime",
        dest="idfy__id_format",
    )
    group.add_argument(
        "-l",
        "--lower-threshold",
        help="lower threshold for feature identification",
        metavar="float",
        type=float,
        dest="idfy__lower_threshold",
    )
    group.add_argument(
        "--lower-thresholds-monthly",
        help="monthly lower thresholds for feature identification",
        metavar="float",
        type=float,
        nargs=12,
        dest="idfy__lower_threshold",
    )
    group.add_argument(
        "-u",
        "--upper-threshold",
        help="upper threshold for feature identification",
        metavar="float",
        type=float,
        dest="idfy__upper_threshold",
    )
    group.add_argument(
        "--minsize",
        help="minimum feature size in number of pixels",
        metavar="n",
        type=int,
        default=1,
        dest="idfy__minsize",
    )
    group.add_argument(
        "--maxsize",
        help="maximum feature size in number of pixels",
        metavar="n",
        type=int,
        default=-1,
        dest="idfy__maxsize",
    )
    group.add_argument(
        "--minsize-km2",
        help="minimum feature size in km^2",
        metavar="n",
        type=int,
        default=0,
        dest="idfy__minsize_km2",
    )
    group.add_argument(
        "--maxsize-km2",
        help="maximum feature size in km^2",
        metavar="n",
        type=int,
        default=-1,
        dest="idfy__maxsize_km2",
    )
    group.add_argument(
        "--grow-features",
        help="grow features by N grid points",
        metavar="N",
        type=int,
        default=0,
        dest="idfy__grow_features_n",
    )

    return group


def parser_add_group__split(parser):
    group = parser.add_argument_group("splitting")

    group.add_argument(
        "--split-levels",
        help="levels for regiongrow-based splitting",
        metavar="float",
        nargs="+",
        type=float,
        dest="split__levels",
    )
    group.add_argument(
        "--split-seed-minsize",
        help="min. size (pixels) of seed regions for splitting",
        metavar="int",
        type=int,
        default=1,
        dest="split__seed_minsize",
    )
    group.add_argument(
        "--split-seed-minstrength",
        help="min. strength of seed regions for splitting",
        metavar="float",
        type=float,
        default=0,
        dest="split__seed_minstrength",
    )

    return group


def parser_add_group__exe(parser):
    group = parser.add_argument_group("execution")

    group.add_argument(
        "--par",
        "--parallel",
        help="run parallelized",
        action="store_true",
        default=True,
        dest="exe__parallelize",
    )
    group.add_argument(
        "--seq",
        "--sequential",
        help="run sequentially",
        action="store_false",
        default=True,
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
        "-p",
        "--profiling",
        help="profile identifiation (input/output excluded)",
        action="store_true",
        dest="exe__profiling",
    )
    group.add_argument(
        "--timings",
        help="measure timings (e.g. tot, IO, core)",
        action="store_true",
        default=True,
        dest="exe__timings_measure",
    )
    group.add_argument(
        "--timings-log",
        help="write timings to logfile (.txt)",
        metavar="file",
        dest="exe__timings_logfile",
    )

    return group


def preproc_args(parser, kwas):

    # Collect argument groups
    for key in ["in", "preproc", "out", "topo", "idfy", "split", "exe"]:
        kwas["conf_" + key] = extract_args(kwas, key)

    preproc_args__in__base(parser, kwas, kwas["conf_in"])
    preproc_args__in__add(parser, kwas, kwas["conf_in"])
    preproc_args__out(parser, kwas, kwas["conf_out"])
    preproc_args__preproc(parser, kwas, kwas["conf_preproc"])
    preproc_args__topo(parser, kwas, kwas["conf_topo"])
    preproc_args__idfy(parser, kwas, kwas["conf_idfy"])
    preproc_args__split(parser, kwas, kwas["conf_split"])
    preproc_args__exe(parser, kwas, kwas["conf_exe"])


def preproc_args__in__base(parser, kwas, conf):

    # Prepare timesteps
    try:
        conf["timesteps"] = TimestepGenerator.from_args(
            conf.pop("tss"), conf.pop("ts_ranges")
        )
    except ValueError as e:
        parser.error("error preparing timesteps: " + str(e))

    # Construct input files
    infiles_tss = TimestepStringFormatter(conf.pop("infile_format")).run(
        conf["timesteps"]
    )
    conf["infiles_tss"] = infiles_tss


def preproc_args__in__add(parser, kwas, conf):

    # Copy arguments from other config groups
    conf["minsize"] = kwas["conf_idfy"]["minsize"]
    conf["maxsize"] = kwas["conf_idfy"]["maxsize"]
    conf["minsize_km2"] = kwas["conf_idfy"]["minsize_km2"]
    conf["maxsize_km2"] = kwas["conf_idfy"]["maxsize_km2"]

    # Collect front computation arguments
    conf["conf_comp_fronts"] = {
        k[len("comp_fronts__") :]: conf.pop(k)
        for k in [k for k in conf.keys()]
        if k.startswith("comp_fronts__")
    }
    preproc_args__comp_fronts(parser, kwas, conf["conf_comp_fronts"])

    # Determine whether to reduce grid resolution
    conf["reduce_grid_resolution"] = conf["reduce_grid_stride"] > 1


def preproc_args__preproc(parser, kwas, conf):

    # Check masking arguments
    if conf["mask_varname"]:

        if conf["mask_threshold_lt"] is None and conf["mask_threshold_lt"]:
            parser.error("must pass at least one threshold for masking")

        elif conf["mask_threshold_gt"] is None:
            parser.error("currently only --mask-gt is supported")


def preproc_args__out(parser, kwas, conf):

    # Construct output files
    outfiles_ts = TimestepStringFormatter(conf.pop("outfile_format")).run(
        kwas["conf_in"]["timesteps"]
    )
    conf["outfiles_ts"] = outfiles_ts


def preproc_args__topo(parser, kwas, conf):
    pass


def preproc_args__idfy(parser, kwas, conf):

    lower_threshold = conf.pop("lower_threshold")
    upper_threshold = conf.pop("upper_threshold")
    if "cyclones" not in kwas["conf_in"]["comp_special"]:
        if (lower_threshold, upper_threshold) == (None, None):
            parser.error("must pass lower and/or upper threshold")
    if lower_threshold is not None and upper_threshold is not None:
        assert len(lower_threshold) == len(upper_threshold)
    elif lower_threshold is not None:
        upper_threshold = -1
    elif upper_threshold is not None:
        lower_threshold = -1
    conf["thresholds"] = [lower_threshold, upper_threshold]

    # Check type codes; supply defaults if omitted
    type_code = conf["type_code"]
    feature_name = kwas["conf_out"]["feature_name"]
    if type_code is not None:
        if type_code < 0 or type_code > 999:
            parser.error(
                ("invalid type code {} for {}; must be in range " "[0, 999]").format(
                    feature_name, type_code
                )
            )
            raise ValueError(err)
    else:
        try:
            type_code = DEFAULT_TYPE_CODES[feature_name]
        except KeyError:
            parser.error(
                (
                    "no default feature type code found for {}; please "
                    "specify type codes or add new default"
                ).format(feature_name)
            )
    conf["type_code"] = type_code


def preproc_args__split(parser, kwas, conf):

    # SR_TMP<
    split_levels = conf["levels"]
    if split_levels and len(split_levels) > 1:
        parser.error(
            (
                "cannot accept split levels: {}\n\n"
                "splitting with multiple split levels resulted in segfault "
                "the last time I've tried, so it's currently disabled\n\n"
                "command which segfaulted:\n\n./bin.dev/identify_features.py "
                "-v=TOT_PREC -n=prec10 -i=data.lm_f.2/{YYYY}/{MM}/lffd{YYYY}"
                "{MM}{DD}{HH}.nc -s 20070101{00..23} -o=precip/prec01_split-"
                "0.5-1.0-3000_2007${m}${d} -t 0.1 -1 --split-levels 0.5 1.0 "
                "--split-seed-minsize 3000\n"
            ).format(thresholds)
        )
    # SR_TMP>


def preproc_args__exe(parser, kwas, conf):

    if conf["profiling"] and conf["parallelize"]:
        parser.error("profiling only supported for sequential execution")

    if conf["timings_logfile"] is not None:
        conf["timings_measure"] = True


def cli():
    parser = setup_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    kwas = vars(parser.parse_args())
    print_args(kwas)
    preproc_args(parser, kwas)
    main(**kwas)


if __name__ == "__main__":
    cli()
