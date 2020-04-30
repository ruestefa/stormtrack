#!/usr/bin/env python3

# Standard library
import argparse
import functools
import json
import logging as log
import multiprocessing as mp
import os
import pstats
import cProfile
import re
import sys
from copy import copy
from pprint import pprint
from pprint import pformat
from time import sleep
from timeit import default_timer as timer

# Third-party
import numpy as np
import pathlib

# Local
from .utils.various import ipython
from .utils.various import extract_args
from .utils.various import print_args
from .utils.various import TimestepGenerator
from .utils.various import TimestepStringFormatter
from .core.typedefs import default_constants
from .core.typedefs import Grid
from .core.tracking import FeatureTracker
from .core.io import read_feature_file
from .core.io import read_feature_files
from .core.io import write_feature_file
from .utils.spatial import derive_lonlat_1d
from .identify_features import read_lonlat
from .identify_features import identify_features as identify_features_core
from .identify_features import read_topo
from .identify_features import get_info_features

log.basicConfig(format="%(message)s", level=log.INFO)
# log.getLogger().setLevel(log.DEBUG)


JSON_BLOCK_ORDER = ["CONFIG", "TRACKS", "EVENTS", "FEATURES"]


def timkey(name):
    return "{}_{}".format(name, mp.current_process())


timings = {}


def main(
    conf_in,
    conf_out,
    conf_tracker,
    conf_track_pp,
    conf_exe,
    conf_preproc,
    conf_topo,
    conf_idfy,
    conf_split,
):

    log.info("Hello fellow feature tracker!")

    global timings

    if conf_exe["timings_measure"]:
        timings["tot"] = mp.Value("d", 0.0)
        timings[timkey("tot_start")] = timer()

        timings["input"] = mp.Value("d", 0.0)
        timings[timkey("input_start")] = timer()

        timings["output"] = mp.Value("d", 0.0)
        timings["core"] = mp.Value("d", 0.0)
        timings["wait"] = mp.Value("d", 0.0)

    # Read lon/lat
    lon2d, lat2d = read_lonlat(conf_in)
    nx, ny = lon2d.shape
    print("GRID : {} x {}".format(nx, ny))
    conf_in["nx"] = nx
    conf_in["ny"] = ny
    conf_tracker["nx"] = nx
    conf_tracker["ny"] = ny
    lon1d, lat1d = derive_lonlat_1d(lon2d, lat2d, nonreg_ok=True)

    # Read topography (if necessary)
    read_topo(conf_topo, conf_in)

    if conf_exe["timings_measure"]:
        with timings["input"].get_lock():
            timings["input"].value += timer() - timings.pop(timkey("input_start"))

    print_conf("CONF_IDENTIFY", conf_idfy)

    # Set up and print configuration
    info_tracks = conf_tracker.copy()
    print_conf("CONF_TRACKER", conf_tracker)

    # Add lon/lat arrays
    conf_tracker["lon1d"] = lon1d
    conf_tracker["lat1d"] = lat1d
    conf_tracker["lon2d"] = lon2d
    conf_tracker["lat2d"] = lat2d
    if conf_tracker["size_lonlat"]:
        # SR_TMP<
        if lon1d is None or lat1d is None:
            err = (
                "cannot compute Feature.area_lonlat (--size-lonlat) on "
                "non-regular grid (lon1d/lat1d is None); need to "
                "implement precise grid-based (method='grid') function "
                "to compute the feature area in km^2 as the currently "
                "available pyproj-based (method='proj') solution is "
                "far too imprecise to be used in practise!"
            )
            raise NotImplementedError(err)
        # SR_TMP>

    # Enable profiling
    if conf_exe["profile"]:
        pr = cProfile.Profile()
        pr.enable()

    # Initialize grid
    if conf_exe["overlay_identification_tracking"]:
        grid = None
    else:
        const = default_constants(nx=nx, ny=ny)
        grid = Grid(const)

    # Prepare input and output files
    infiles_tss_sorted = sorted(conf_in["infiles_tss"].items())
    outfiles_tss_sorted = sorted(conf_out["outfiles_tss"].items())
    infiles_tss_per_outfile = {}
    for timesteps_out, outfile in outfiles_tss_sorted:
        infiles_tss_per_outfile[outfile] = []
        for timesteps_in, infile in infiles_tss_sorted:
            if all(ts in timesteps_out for ts in timesteps_in):
                infiles_tss_per_outfile[outfile].append((timesteps_in, infile))
            elif any(ts in timesteps_out for ts in timesteps_in):
                err = (
                    "input period ({}..{}) only partly overlaps output "
                    "period ({}..{}); must be even subperiods!"
                ).format(
                    min(timesteps_in),
                    max(timesteps_in),
                    min(timesteps_out),
                    max(timesteps_out),
                )
                raise Exception(err)
            else:
                continue

    # Initialize tracker, etc.
    tracker = FeatureTracker(**conf_tracker)
    conf_out["nout"] = len(outfiles_tss_sorted)
    finished = []
    pending = []
    nts = len(conf_in["timesteps"])
    timestep_prev = mp.Value("L", 0)

    if conf_in["restart"]:
        # Restart tracker
        conf_out["iout"] = restart_tracker(
            tracker,
            conf_in["restart_files_in_tss"],
            conf_in["restart_files_out_tss"],
            conf_in["feature_name"],
            finished,
            pending,
            conf_out["link_features"],
            conf_out["linked_files_tss"],
        )
    else:
        conf_out["iout"] = 0

    outfiles = dict(curr=None, next=None)
    infos_features = mp.Manager().dict(curr=None, prev=None)

    # Run tracking
    its_done = mp.Value("i", 0)
    for timesteps_out, _outfile_next in outfiles_tss_sorted:
        outfiles["next"] = _outfile_next
        infiles_tss_sorted_i = infiles_tss_per_outfile[_outfile_next]

        if not conf_exe["overlay_identification_tracking"]:
            # Run identification and tracking one after the other
            # at each sequential timestep
            for timesteps_in, infile in infiles_tss_sorted_i:

                features_ts = {}

                run_feature_input(
                    features_ts,
                    infos_features,
                    grid,
                    lon2d,
                    lat2d,
                    conf_in,
                    conf_out,
                    conf_topo,
                    conf_preproc,
                    conf_idfy,
                    conf_split,
                    conf_exe,
                    timesteps_in,
                    infile,
                )

                run_tracking_timesteps(
                    tracker,
                    finished,
                    pending,
                    features_ts,
                    infos_features,
                    info_tracks,
                    outfiles,
                    nts,
                    lon2d,
                    lat2d,
                    grid,
                    conf_in,
                    conf_out,
                    conf_topo,
                    conf_preproc,
                    conf_idfy,
                    conf_split,
                    conf_track_pp,
                    conf_exe,
                    its_done,
                    timestep_prev,
                    timesteps_in,
                )

        else:
            #
            # Run identification and tracking in parallel
            #
            # The identification runs continuously until the features at all
            # timesteps have been (read from disk or) identified, while
            # the tracking increments by one timestep as soon as the
            # respective features are available
            #
            # This assumes that the identification takes similarly long or
            # longer than the tracking; if the identification is much faster,
            # features might pile up and use a lot of memory
            #
            features_ts = mp.Manager().dict()

            # Identify features
            fct_id = _run_feature_input__tss
            args_id = (
                infos_features,
                grid,
                lon2d,
                lat2d,
                conf_in,
                conf_out,
                conf_topo,
                conf_preproc,
                conf_idfy,
                conf_split,
                conf_exe,
                features_ts,
                infiles_tss_sorted_i,
                conf_exe["num_procs_id"],
            )
            p_id = mp.Process(target=fct_id, args=args_id)
            p_id.start()

            # Track features
            _run_feature_tracking__tss(
                tracker,
                finished,
                pending,
                features_ts,
                infos_features,
                info_tracks,
                outfiles,
                nts,
                lon2d,
                lat2d,
                grid,
                conf_in,
                conf_out,
                conf_topo,
                conf_preproc,
                conf_idfy,
                conf_split,
                conf_track_pp,
                conf_exe,
                its_done,
                timestep_prev,
                infiles_tss_sorted_i,
            )

            p_id.join()

    # Fetch last timestep (for output)
    timestep = timesteps_in[-1]

    # Force-finish all remaining active tracks
    if conf_exe["timings_measure"]:
        timings[timkey("core_start")] = timer()
    finish_tracks(tracker, finished, conf_track_pp)
    if conf_exe["timings_measure"]:
        with timings["core"].get_lock():
            timings["core"].value += timer() - timings.pop(timkey("core_start"))

    # Write remaining tracks to disk
    if finished:
        pending_i = dict(
            ts=timestep_prev.value, file=outfiles["curr"], iout=conf_out["iout"],
        )
        pending.append(pending_i)

        if conf_exe["timings_measure"]:
            timings[timkey("output_start")] = timer()
        for pending_i in pending:
            nx, ny = lon2d.shape
            write_pending_tracks(
                finished,
                infos_features["curr"],
                info_tracks,
                nx,
                ny,
                conf_out,
                timestep,
                pending_i,
            )
        if conf_exe["timings_measure"]:
            with timings["output"].get_lock():
                timings["output"].value += timer() - timings.pop(timkey("output_start"))

    if conf_exe["timings_measure"]:
        timings["tot"].value = timer() - timings.pop(timkey("tot_start"))

        print("\n timings:")
        for key, val in sorted(timings.items()):
            print(" {:20} : {:10.3f}s".format(key, val.value))

        print("")
        if conf_exe["timings_logfile"]:
            print("write timings to {}".format(conf_exe["timings_logfile"]))
            with open(conf_exe["timings_logfile"], "w") as fo:
                for key, val in sorted(timings.items()):
                    fo.write(" {:20} {:10.3f}\n".format(key, val.value))
            print("")

    if conf_exe["profile"]:
        pr.disable()
        ps = pstats.Stats(pr)
        nlines = conf_exe["profile_nlines"]
        for psort in conf_exe["psort_list"]:
            ps.strip_dirs().sort_stats(psort).print_stats(nlines)


# Wrapper functions for parallel input/tracking


def _run_feature_input__tss(
    infos_features,
    grid,
    lon2d,
    lat2d,
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_exe,
    features_ts,
    infiles_tss_sorted_i,
    num_procs_id,
):
    fct = functools.partial(
        _run_feature_input__tss__seq,
        infos_features,
        grid,
        lon2d,
        lat2d,
        conf_in,
        conf_out,
        conf_topo,
        conf_preproc,
        conf_idfy,
        conf_split,
        conf_exe,
        features_ts,
    )

    if num_procs_id == 1:
        # Run sequentially
        fct(infiles_tss_sorted_i)

    else:
        # Run in parallel
        args_grouped = [[] for _ in range(num_procs_id)]
        for i, args in enumerate(infiles_tss_sorted_i):
            j = i % num_procs_id
            args_grouped[j].append(args)

        pool = mp.Pool(num_procs_id, maxtasksperchild=1)
        pool.map(fct, args_grouped)


def _run_feature_input__tss__seq(
    infos_features,
    grid,
    lon2d,
    lat2d,
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_exe,
    features_ts,
    infiles_tss_sorted_i,
):
    for timesteps_i, infile in infiles_tss_sorted_i:
        run_feature_input(
            features_ts,
            infos_features,
            grid,
            lon2d,
            lat2d,
            conf_in,
            conf_out,
            conf_topo,
            conf_preproc,
            conf_idfy,
            conf_split,
            conf_exe,
            timesteps_i,
            infile,
        )


def _run_feature_tracking__tss(
    tracker,
    finished,
    pending,
    features_ts,
    infos_features,
    info_tracks,
    outfiles,
    nts,
    lon2d,
    lat2d,
    grid,
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_track_pp,
    conf_exe,
    its_done,
    timestep_prev,
    infiles_tss_sorted_i,
):
    for timesteps_in, infile in infiles_tss_sorted_i:
        run_tracking_timesteps(
            tracker,
            finished,
            pending,
            features_ts,
            infos_features,
            info_tracks,
            outfiles,
            nts,
            lon2d,
            lat2d,
            grid,
            conf_in,
            conf_out,
            conf_topo,
            conf_preproc,
            conf_idfy,
            conf_split,
            conf_track_pp,
            conf_exe,
            its_done,
            timestep_prev,
            timesteps_in,
        )


def run_tracking_timesteps(
    tracker,
    finished,
    pending,
    features_ts,
    infos_features,
    info_tracks,
    outfiles,
    nts,
    lon2d,
    lat2d,
    grid,
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_track_pp,
    conf_exe,
    its_done,
    timestep_prev,
    timesteps_in,
):

    for its_curr, timestep in enumerate(timesteps_in):
        its = its_done.value + its_curr

        run_tracking_timestep(
            tracker,
            finished,
            pending,
            features_ts,
            infos_features,
            info_tracks,
            outfiles,
            its,
            nts,
            lon2d,
            lat2d,
            grid,
            conf_in,
            conf_out,
            conf_topo,
            conf_preproc,
            conf_idfy,
            conf_split,
            conf_track_pp,
            conf_exe,
            timestep,
            timestep_prev,
        )

        timestep_prev.value = timestep

    its_done.value += len(timesteps_in)


def run_tracking_timestep(
    tracker,
    finished,
    pending,
    features_ts,
    infos_features,
    info_tracks,
    outfiles,
    its,
    nts,
    lon2d,
    lat2d,
    grid,
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_track_pp,
    conf_exe,
    timestep,
    timestep_prev,
):

    global timings

    ts0 = tracker.min_ts_start()
    msg = "[{}-{}] {:4} finished, {:4} active".format(
        (ts0 if ts0 is not None else timestep),
        timestep,
        len(finished),
        len(tracker.active_tracks),
    )
    if pending:
        msg += "; {} pending output".format(len(pending))
        if len(pending) == 1:
            msg += ": {}".format(pending[0]["ts"])
        else:
            msg += "s: {}{}{}".format(
                pending[0]["ts"],
                (", " if len(pending) == 2 else ".."),
                pending[-1]["ts"],
            )
    log.info(msg)

    if conf_exe["timings_measure"]:
        timings[timkey("core_start")] = timer()

    if is_time_to_split_active_tracks(its, conf_track_pp):
        # Periodically Split active tracks to avoid buildup of
        # huge tracks which are are split up eventually, anyway
        split_active_tracks(tracker, finished, timestep, conf_track_pp)

    if conf_exe["timings_measure"]:
        with timings["core"].get_lock():
            timings["core"].value += timer() - timings.pop(timkey("core_start"))

    if conf_exe["timings_measure"]:
        timings[timkey("output_start")] = timer()

    # Check if output is ready
    if outfiles["curr"] is None:
        outfiles["curr"] = outfiles["next"]
    if outfiles["next"] != outfiles["curr"]:

        # Postpone writing output until there are no more active tracks
        # that start in the output period (e.g., duration criteria can
        # not be properly applied otherwise because the total duration
        # of a track is not known as long it is active)
        _pending_i = dict(
            ts=timestep_prev.value, file=outfiles["curr"], iout=conf_out["iout"]
        )
        pending.append(_pending_i)

        outfiles["curr"] = outfiles["next"]
        conf_out["iout"] += 1

    info_features = infos_features["curr"]

    # Write tracks to disk
    min_ts_start = tracker.min_ts_start()
    for pending_i in [p for p in pending]:
        if min_ts_start is not None and min_ts_start <= pending_i["ts"]:
            break
        pending.remove(pending_i)
        nx, ny = lon2d.shape
        write_pending_tracks(
            finished, info_features, info_tracks, nx, ny, conf_out, timestep, pending_i
        )

    if conf_exe["timings_measure"]:
        with timings["output"].get_lock():
            timings["output"].value += timer() - timings.pop(timkey("output_start"))

    if conf_exe["timings_measure"]:
        timings[timkey("wait_start")] = timer()

    # Fetch features
    wait_s = 2
    iter_max = 2 * 3600
    for iter_i in range(iter_max):
        try:
            features = features_ts.pop(timestep)
        except KeyError:
            # Features not yet available, so wait and try again
            msg = "[{}] waiting : {} s".format(timestep, iter_i * wait_s)
            try:
                w, h = os.get_terminal_size()
            except OSError:
                pass
            else:
                print(msg[:w], end="\r", flush=True)
            sleep(wait_s)
        else:
            # Features available, so we're good to go!
            break
    else:
        err = (
            "TIMEOUT ({:,} s) waiting for features at timestep {}\n"
            "This is usually caused by either missing input files, "
            "or by an error in an identification sub-process; "
            "scroll upward to check for an earlier exception"
        ).format(wait_s * iter_max, timestep)
        raise Exception(err)

    if conf_exe["timings_measure"]:
        with timings["wait"].get_lock():
            timings["wait"].value += timer() - timings.pop(timkey("wait_start"))

    if conf_exe["timings_measure"]:
        timings[timkey("core_start")] = timer()

    # Track the features
    track_features(tracker, features, timestep, finished, conf_track_pp)

    if conf_exe["timings_measure"]:
        with timings["core"].get_lock():
            timings["core"].value += timer() - timings.pop(timkey("core_start"))


def check_info_set_prev(infos):
    if infos["prev"] is not None:
        if infos["prev"] != infos["curr"]:
            err = ("info dicts differ:\n\nNEW:\n{}\n\nOLD:\n{}\n").format(
                pformat(sorted(infos["curr"].items())),
                pformat(sorted(infos["prev"].items())),
            )
            raise Exception(err)
    infos["prev"] = infos["curr"].copy()


def print_conf(name, conf):
    """Print the configuration dict."""
    print("\n" + "-" * 60)
    print(" {} :".format(name))
    print("" + "-" * 60)
    # for key, val in conf.items():
    for key, val in sorted(conf.items()):
        print("{:40} : {}".format(key, val))
    print("-" * 60 + "\n")


def restart_tracker(
    tracker,
    infiles_tss,
    outfiles_tss,
    feature_name,
    finished,
    pending,
    link_features,
    linked_files_tss,
):
    """Restart tracker from unfinished tracking run.

    Because separate unfinished tracks in the existing files might be merged,
    it is necessary to rewrite the track files used for the restart.

    If tracks are shorter than one output period (e.g., for fronts stored
    in monthly files), only the most recent track file needs to be read
    and rewritten.

    If tracks span more than two input files, it is necessary to read and
    rewrite all files that contain a piece of any unfinished track.

    If features are only linked to tracks and not written to the track files,
    it is only necessary to read/rebuild the pixels for the most recent set
    of features, as only these will be necessary to continue the tracks with
    new features. For all previous tracks, the feature pixels can be omitted,
    which results in faster input.
    """

    # Sort restart files in reverse chronological order
    infiles_tss_sorted = sorted(infiles_tss.items(), reverse=True)

    # Read first restart file (or last, chronologically)
    # Rebuild feature pixels in any case (needed to continue tracks)
    infile = infiles_tss_sorted[0][1]
    print("tracker restart: read {}".format(infile))
    _r = read_feature_file(
        infile, feature_name=feature_name, counter=True, ignore_missing_neighbors=True
    )
    jdat = _r["jdat"]
    tracks = _r["tracks"]

    # Check tracker setup
    tracker.check_setup(jdat["info_tracks"], action="raise")

    # Determine head timestep
    ts_end = max(infiles_tss_sorted[0][0])

    # Determine earliest start of any unfinished track
    ts_start = None
    for track in tracks:
        if not track.is_complete() and ts_end in track.timesteps():
            ts_start_i = min(track.timesteps(total=True))
            if ts_start is None or ts_start_i < ts_start:
                ts_start = ts_start_i

    # Collect necessary track files and initialie pending output
    pending.clear()
    iout = 0
    infiles_todo = []
    for tss, infile in infiles_tss_sorted[1:]:
        infiles_todo.append(infile)
        outfile = outfiles_tss[tss]
        pending.append(dict(ts=max(tss), file=outfile, iout=iout))
        if link_features:
            linked_files_tss[outfile] = get_linked_feature_files(infile)
        iout += 1
        if ts_start in tss:
            break
    else:
        err = "restart timestep period not long enough: {} not found".format(ts_start)
        raise Exception(err)

    # Read remaining track files
    print("read remaining {} restart files".format(len(infiles_todo)))
    _r = read_feature_files(
        infiles_todo,
        feature_name=feature_name,
        extra_tracks=tracks,
        ignore_missing_neighbors=True,
        is_subperiod=True,
        counter=True,
        counter_core=True,
    )
    tracks = _f["tracks"]

    # Restart tracker
    tracker.restart(tracks, ts_end)
    finished.extend(tracker.pop_finished_tracks())

    return iout


def get_linked_feature_files(infile):
    with open(infile, "r") as fi:
        jdat = json.load(fi)
    if not jdat["header"].get("link_features"):
        raise Exception("track file has no linked features: {}".format(infile))
    return jdat["header"]["feature_files_tss"]


def run_feature_input(
    features_ts,
    infos_features,
    grid,
    lon2d,
    lat2d,
    conf_in,
    conf_out,
    conf_topo,
    conf_preproc,
    conf_idfy,
    conf_split,
    conf_exe,
    timesteps_in,
    infile,
):

    global timings

    if conf_exe["timings_measure"]:
        timings[timkey("input_start")] = timer()

    # Read/identify features
    if not conf_in["identify_features"]:

        # Read precomputed features
        features, infos_features["curr"] = import_features(
            infile, timesteps_in, conf_in
        )
        print("read {} features from {}".format(len(features), infile))
        for ts in timesteps_in:
            features_ts[ts] = [f for f in features if f.timestep == ts]
        check_info_set_prev(infos_features)

    else:
        for timestep in timesteps_in:

            # Identify new features at the current timestep
            timings_measure = False  # SR_TMP
            features, timings_idfy = identify_features_core(
                name=conf_out["feature_name"],
                conf_in=conf_in,
                conf_topo=conf_topo,
                conf_preproc=conf_preproc,
                conf_idfy=conf_idfy,
                conf_split=conf_split,
                lon=lon2d,
                lat=lat2d,
                timestep=timestep,
                timings_measure=timings_measure,
                grid=grid,
                silent=True,
            )
            infos_features["curr"] = get_info_features(
                conf_in, conf_idfy, conf_split, conf_topo
            )
            check_info_set_prev(infos_features)

            features_ts[timestep] = features

    if conf_exe["timings_measure"]:
        with timings["input"].get_lock():
            timings["input"].value += timer() - timings.pop(timkey("input_start"))


def import_features(infile, timesteps, conf_in):
    debug = False

    # Read features
    feature_name = conf_in["feature_name"]
    minsize = conf_in["minsize"]
    maxsize = conf_in["maxsize"]
    _r = read_feature_file(
        infile,
        feature_name=feature_name,
        minsize=minsize,
        maxsize=maxsize,
        ignore_missing_neighbors=True,
        silent=~debug,
    )
    features = _r["features"]
    jdat = _r["jdat"]

    features = [f for f in features if f.timestep in timesteps]

    name_info = "{}_{}".format("info", feature_name)
    info = jdat[name_info]

    # Dirty check for unique feature ids (sure cannot hurt)
    assert len(set([c.id for c in features])) == len(features)

    return features, info


def write_pending_tracks(
    finished, info_features, info_tracks, nx, ny, conf_out, timestep, pending
):

    # Split tracks into fragments for output
    tracks_out = []
    for track in [t for t in finished]:
        if track.ts_end(total=False) <= pending["ts"]:
            finished.remove(track)
            tracks_out.append(track)
        elif track.ts_start(total=False) <= pending["ts"]:
            fragment = track.cut_off(until=pending["ts"])
            # SR_DBG<
            if fragment.id != track.id:
                err = "error splitting track at outfile boundary"
                raise Exception(err)
            # SR_DBG>
            tracks_out.append(fragment)

    # Write tracks to disk
    write_tracks(
        pending["file"],
        finished=tracks_out,
        info_features=info_features,
        info_tracks=info_tracks,
        nx=nx,
        ny=ny,
        timestep=pending["ts"],
        iout=pending["iout"],
        conf_out=conf_out,
    )


def write_tracks(
    outfile, *, finished, info_features, info_tracks, nx, ny, timestep, iout, conf_out
):

    log.info("+" * 60)
    if (
        conf_out["skip_output"]
        or iout < conf_out["output_skip_start"]
        or iout >= (conf_out["nout"] - conf_out["output_skip_end"])
    ):
        # Skip output for this step; dismiss the respective tracks
        log.info("skip output for timestep {}".format(timestep))

    else:
        # Write output to disk
        log.info(
            "save {} tracks ({} features) to {}".format(
                len(finished), conf_out["feature_name"], outfile
            )
        )
        write_feature_file(
            outfile,
            tracks=finished,
            feature_name=conf_out["feature_name"],
            info_tracks=info_tracks,
            info_features=info_features,
            nx=nx,
            ny=ny,
            track_store_mode=conf_out["track_store_mode"],
            pixel_store_mode=conf_out["pixel_store_mode"],
            link_features=conf_out["link_features"],
            feature_files_tss=conf_out["linked_files_tss"][outfile],
        )
    log.info("+" * 60)

    finished.clear()


def track_features(tracker, features, timestep, finished, conf_track_pp):
    """Start, extend, and finish existing tracks with a new set of features.

    Return all newly finished tracks.
    """
    debug = False

    # Start/finish/extend tracks with new set of features
    tracker.extend_tracks(features, timestep)

    # Get all newly finished tracks
    tracks = tracker.pop_finished_tracks()
    nold = len(tracks)
    if debug:
        log.debug("finished {} tracks".format(nold))

    # Remove short tracks and segments etc.
    tracks = postproc_tracks(tracks, tracker, conf_track_pp)

    finished.extend(tracks)


def is_time_to_split_active_tracks(its, conf_track_pp):
    interval = conf_track_pp["split_active_tracks_interval"]
    return its > 0 and its % interval == 0


def split_active_tracks(tracker, finished, timestep, conf_track_pp):
    """Split active tracks to prevent them from growing huge.

    Tracks are usually only split once they're finished. However, some
    branchings can already be confidently eliminated even while the track
    is still active, which prevents it from growing huge and slowing the
    tracking down (only to be eventually split up anyway).
    """
    nsplit = tracker.split_active_tracks()
    if nsplit == 0:
        return

    tracks = tracker.pop_finished_tracks()
    log.info(
        "[{}-{}] active track splitting: finish {} tracks".format(
            tracker.min_ts_start(), timestep, len(tracks)
        )
    )

    tracks = postproc_tracks(tracks, tracker, conf_track_pp)

    finished.extend(tracks)


def finish_tracks(tracker, finished, conf_track_pp):
    debug = False

    tracker.finish_tracks()
    tracks = tracker.pop_finished_tracks()
    if debug:
        log.debug("finished {} tracks".format(len(tracks)))

    tracks = postproc_tracks(tracks, tracker, conf_track_pp)
    finished.extend(tracks)


def postproc_tracks(tracks, tracker, conf_track_pp):
    """Post-process finished tracks (e.g. remove stubs)."""
    debug = False

    nold = len(tracks)

    # Removed too short tracks
    min_dur = conf_track_pp["min_duration"]
    tracks = [t for t in tracks if t.duration(total=True) >= min_dur]
    if debug:
        log.debug("remove short tracks: {} -> {}".format(nold, len(tracks)))

    log.debug("postproc: {} tracks -> {} tracks".format(nold, len(tracks)))
    return tracks


from .identify_features import parser_extend_group__in
from .identify_features import parser_add_group__preproc
from .identify_features import parser_add_group__topo
from .identify_features import parser_add_group__idfy
from .identify_features import parser_add_group__split
from .identify_features import preproc_args__in__add
from .identify_features import preproc_args__preproc
from .identify_features import preproc_args__topo
from .identify_features import preproc_args__idfy
from .identify_features import preproc_args__split


description = """Track features"""


def setup_parser():
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input = parser_add_group__in(parser)
    parser_add_group__out(parser)
    parser_add_group__tracker(parser)
    parser_add_group__track_pp(parser)
    parser_add_group__exe(parser)

    # From identify_features.py
    parser_extend_group__in(parser, input)
    parser_add_group__preproc(parser)
    parser_add_group__topo(parser)
    parser_add_group__idfy(parser)
    parser_add_group__split(parser)

    return parser


def parser_add_group__in(parser):
    group = parser.add_argument_group("input")

    group.add_argument(
        "-i",
        "--infile-fmt",
        help=(
            "template of input file names used to extract the "
            "timestep, including (relative) path to the files "
            "(e.g. foo/bar_{YYYY}{MM}{DD}_{HH}.json); either raw "
            "input (netcdf) or precomputed features (json)"
        ),
        metavar="format-string",
        dest="in__infile_fmt",
    )
    group.add_argument(
        "-s",
        "--timesteps",
        help="timesteps (format: e.g., YYYYMMDDHH)",
        metavar="timestep",
        type=int,
        nargs="+",
        dest="in__tss",
    )
    group.add_argument(
        "-S",
        "--timesteps-range",
        help="timestep range (format: e.g., YYYYMMDDHH)",
        metavar=("start", "end", "stride"),
        nargs=3,
        type=int,
        dest="in__ts_range",
    )
    group.add_argument(
        "-n",
        "--feature-name",
        help="name of front features in input file",
        metavar="name",
        default="tracking",
        dest="in__feature_name",
    )
    group.add_argument(
        "--infile-lonlat",
        help="lonlat file (.npz)",
        metavar="file",
        default="crclim_lonlat.npz",
        dest="in__infile_lonlat",
    )
    group.add_argument(
        "--lonlat-names",
        help="names of lon and lat variables (lon/lat or rlon/rlat)",
        nargs=2,
        # default = ["rlon", "rlat"],
        default=["lon", "lat"],
        dest="in__lonlat_names",
    )
    group.add_argument(
        "--lonlat-transpose",
        help="transpose lon/lat fields",
        action="store_true",
        dest="in__lonlat_transp",
    )
    group.add_argument(
        "--restart-file-fmt",
        help=(
            "track file (.json) from which to restart tracking " "(with {YYYY} etc.)"
        ),
        metavar="file-fmt",
        dest="in__restart_file_fmt",
    )
    group.add_argument(
        "--restart-timesteps-range",
        help=(
            "restart timestep range (format: e.g., YYYYMMDDHH); "
            "need to pass enough timesteps to cover the full "
            "lifetime of all unfinished tracks because the "
            "respective files need to be read and rewritten; "
            "note that no files will be read and rewritten "
            "unnecessarily, no matter how long the restart "
            "timesteps range (i.e., best be generous"
        ),
        metavar=("start", "end", "stride"),
        nargs=3,
        type=int,
        dest="in__restart_ts_range",
    )

    return group


def parser_add_group__out(parser):
    group = parser.add_argument_group("output")

    group.add_argument(
        "--track-store-mode",
        help="how to store the track and feature object data",
        metavar="mode",
        choices=("json", "graph"),
        default="json",
        dest="out__track_store_mode",
    )
    group.add_argument(
        "--pixel-store-mode",
        help="how to store the feature pixels",
        default="pixels",
        choices=["pixels", "boundaries"],
        metavar="mode",
        dest="out__pixel_store_mode",
    )
    group.add_argument(
        "--link-features",
        help=(
            "link to existing feature file to avoid writing then "
            "to disk; requires precomputed features as input"
        ),
        action="store_true",
        dest="out__link_features",
    )
    group.add_argument(
        "-o",
        "--outfile-fmt",
        help="output file pattern (e.g. foo/bar_{YYYY}{MM}.json)",
        metavar="pattern",
        dest="out__outfile_fmt",
        required=True,
    )
    group.add_argument(
        "--output-skip-start",
        help="skip output N times from the beginning",
        type=int,
        default=0,
        metavar="N",
        dest="out__output_skip_start",
    )
    group.add_argument(
        "--output-skip-end",
        help="skip output N times from the end",
        type=int,
        default=0,
        metavar="N",
        dest="out__output_skip_end",
    )
    group.add_argument(
        "--skip-output",
        help="skip output altogether (e.g., for debugging)",
        action="store_true",
        dest="out__skip_output",
    )
    group.add_argument(
        "--link-existing-feature-files",
        help=(
            "when computing features on-the-fly, don't write "
            "them to disk, but link to existing feature files "
            "(specified with --existing-feature-file-fmt) "
            "to speed up output; useful if the exact same set "
            "of features is tracked many times, e.g., during "
            "sensitivity runs"
        ),
        action="store_true",
        dest="out__link_existing_feature_files",
    )
    group.add_argument(
        "--existing-feature-file-fmt",
        help=(
            "file format for existing feature files "
            "(see --link-existing-feature-files)"
        ),
        metavar="file-fmt",
        dest="out__existing_feature_file_fmt",
    )

    return group


def parser_add_group__tracker(parser):
    group = parser.add_argument_group("tracker setup")

    group.add_argument(
        "--alpha",
        help="parameter alpha (0: area only; 1: overlap-only)",
        metavar="float",
        type=float,
        default=0.5,
        dest="tracker__alpha",
    )
    group.add_argument(
        "--min-p-overlap",
        help="threshold for overlap tracking probability",
        metavar="float",
        type=float,
        default=0.0,
        dest="tracker__min_p_overlap",
    )
    group.add_argument(
        "--min-p-size",
        help="threshold for area tracking probability",
        metavar="float",
        type=float,
        default=0.0,
        dest="tracker__min_p_size",
    )
    group.add_argument(
        "--size-lonlat",
        help="use lon/lat-based feature sizes, not just no. pixels",
        action="store_true",
        dest="tracker__size_lonlat",
    )
    group.add_argument(
        "--min-p-tot",
        help="threshold for total tracking probability'",
        metavar="float",
        type=float,
        default=0.0,
        dest="tracker__min_p_tot",
    )
    group.add_argument(
        "--max-children",
        help="max. no. successors/predecessors (-1: unlimited)",
        metavar="n",
        type=int,
        default=4,
        dest="tracker__max_children",
    )
    group.add_argument(
        "--merge-features",
        help="merge adjacent features of the same track",
        action="store_true",
        default=False,
        dest="tracker__merge_features",
    )
    group.add_argument(
        "--nomerge-features",
        help="don't merge adjacent features of the same track",
        action="store_false",
        default=False,
        dest="tracker__merge_features",
    )
    group.add_argument(
        "--split-tracks",
        help=(
            "parameter N to split tracks at branchings (-1: no "
            "splitting; 0: eliminate all branchings; > 0: "
            "split unless branches remerge in N timesteps window"
        ),
        metavar="N",
        type=int,
        default=-1,
        dest="tracker__split_tracks_n",
    )

    return group


def parser_add_group__track_pp(parser):
    group = parser.add_argument_group("track post-processing")

    group.add_argument(
        "--min-duration",
        help="min. duration for tracks to be kept",
        metavar="n",
        type=int,
        default=1,
        dest="track_pp__min_duration",
    )
    group.add_argument(
        "--split-active-tracks",
        help="split active tracks every N timesteps",
        metavar="N",
        type=int,
        default=24,
        dest="track_pp__split_active_tracks_interval",
    )

    return group


def parser_add_group__exe(parser):
    group = parser.add_argument_group("execution")

    group.add_argument(
        "-p",
        "--profile",
        help=(
            "run cProfile; optionally pass the number of lines "
            "printed in profiling report"
        ),
        type=int,
        metavar="N",
        nargs="*",
        dest="exe__profile_nlines",
    )
    group.add_argument(
        "--psort",
        help="what to sort cProfile output by",
        choices=("ncalls", "tottime", "percall", "cumtime", "percall"),
        default=("tottime", "cumtime"),
        nargs="+",
        dest="exe__psort_list",
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
    group.add_argument(
        "--overlay-id-track",
        help=(
            "overlay identification and tracking: run both in "
            "parallel, with the tracking incrementing one "
            "timestep as soon as the respective features are "
            "available; requires the identification to be "
            "similarly slow as the tracking, or slower; "
            "note that the output is part of the tracking step, "
            "i.e., if the identification is considerably slower "
            "than the tracking, the output can be hidden along"
            "side the tracking"
        ),
        action="store_true",
        dest="exe__overlay_identification_tracking",
    )
    group.add_argument(
        "--num-procs-id",
        help=(
            "parallelize identification; only if identification "
            "and tracking are overlaid; if the identification is "
            "much slower than the tracking, parallelizing it "
            "can further reduce the runtime"
        ),
        metavar="N",
        type=int,
        default=1,
        dest="exe__num_procs_id",
    )

    return group


def preproc_args(parser, kwas):

    # Collect argument groups
    for key in [
        "in",
        "out",
        "tracker",
        "track_pp",
        "exe",
        "preproc",
        "topo",
        "idfy",
        "split",
    ]:
        kwas["conf_" + key] = extract_args(kwas, key)

    preproc_args__in(parser, kwas, kwas["conf_in"])
    preproc_args__out(parser, kwas, kwas["conf_out"])
    preproc_args__tracker(parser, kwas, kwas["conf_tracker"])
    preproc_args__track_pp(parser, kwas, kwas["conf_track_pp"])
    preproc_args__exe(parser, kwas, kwas["conf_exe"])

    # from identify_features.py
    if kwas["conf_in"]["identify_features"]:
        preproc_args__in__add(parser, kwas, kwas["conf_in"])
        preproc_args__preproc(parser, kwas, kwas["conf_preproc"])
        preproc_args__topo(parser, kwas, kwas["conf_topo"])
        preproc_args__idfy(parser, kwas, kwas["conf_idfy"])
        preproc_args__split(parser, kwas, kwas["conf_split"])


def preproc_args__in(parser, kwas, conf):

    # Copy arguments from other config groups
    conf["minsize"] = kwas["conf_idfy"]["minsize"]
    conf["maxsize"] = kwas["conf_idfy"]["maxsize"]
    conf["minsize_km2"] = kwas["conf_idfy"]["minsize_km2"]
    conf["maxsize_km2"] = kwas["conf_idfy"]["maxsize_km2"]

    # Prepare timesteps
    try:
        conf["timesteps"] = TimestepGenerator.from_args(
            conf.pop("tss"), [conf.pop("ts_range")]
        )
    except ValueError as e:
        parser.error("prepare timesteps: {}".format(e))

    # Construct input files
    infile_fmt = conf.pop("infile_fmt")
    conf["infiles_tss"] = TimestepStringFormatter(infile_fmt).run(conf["timesteps"])

    # Check consistency of input and output files in terms of frequency
    outfile_fmt = kwas["conf_out"]["outfile_fmt"]
    for key in ["{YYYY}", "{MM}", "{DD}", "{HH}", "{NN}"]:
        if key in outfile_fmt and key not in infile_fmt:
            parser.error(
                ("inconsistent input and output files: " "{}, {}").format(
                    infile_fmt, outfile_fmt
                )
            )

    conf["identify_features"] = not any(
        infile_fmt.endswith(suffix) for suffix in (".pickle", ".json")
    )

    # Prepare restart
    restart_ts_range = conf.pop("restart_ts_range")
    restart_file_fmt = conf.pop("restart_file_fmt")
    if restart_ts_range and restart_file_fmt:
        conf["restart"] = True
        try:
            restart_tss = TimestepGenerator.from_args(None, [restart_ts_range])
        except ValueError as e:
            parser.error(e)
        conf["restart_files_in_tss"] = TimestepStringFormatter(restart_file_fmt).run(
            restart_tss
        )
        conf["restart_files_out_tss"] = TimestepStringFormatter(
            kwas["conf_out"]["outfile_fmt"]
        ).run(restart_tss)
    elif not restart_ts_range and not restart_file_fmt:
        conf["restart"] = False
        conf["restart_files_tss"] = None
    else:
        parser.error(
            "must pass both --restart-file-fmt and "
            "--restart-timesteps-range to restart tracking"
        )


def preproc_args__out(parser, kwas, conf):

    # Copy arguments from other config groups
    conf["feature_name"] = kwas["conf_in"]["feature_name"]

    # Construct output files
    infiles_tss = kwas["conf_in"]["infiles_tss"]
    outfiles_tss = TimestepStringFormatter(conf.pop("outfile_fmt")).run(
        kwas["conf_in"]["timesteps"]
    )
    conf["outfiles_tss"] = outfiles_tss
    if len(conf["outfiles_tss"]) > len(kwas["conf_in"]["infiles_tss"]):
        parser.error("output frequency cannot exceed input frequency")
    _skip_start = int(conf["output_skip_start"])
    _skip_end = int(conf["output_skip_end"])
    for tss_in in infiles_tss.keys():
        for tss_out in sorted(outfiles_tss.keys())[_skip_start:_skip_end]:
            if any(ts in tss_out for ts in tss_in):
                if not all(ts in tss_out for ts in tss_in):
                    err = "input timesteps must be subsets of output timesteps"
                    parser.error(err)

    # Check linking options (compatibility with other flags)
    if not conf["link_existing_feature_files"]:
        # Make sure features are read if they are to be linked only
        if conf["link_features"] and kwas["conf_in"]["identify_features"]:
            parser.error(
                "incompatible options --link-features and "
                "--identify-features; must input precomputed features to "
                "avoid writing them to disk by linking them"
            )
    else:
        # Linking existing feature files necessitates feature identification
        # (otherwise the normal option to link features should be used)
        if not kwas["conf_in"]["identify_features"]:
            parser.error(
                "incompatible options --link-existing-feature-files "
                "and --infile-fmt=*.json; use --link-features instead "
                "in order to read and link existing features"
            )
        if conf["link_features"]:
            parser.error(
                "incompatible options --link-features and "
                "--link-existing-feature-files; use the former to "
                "read preexisting features and the other to identify "
                "features on-the-fly (which might be faster than reading) "
                "but link to an existing set of identical features"
            )
        conf["link_features"] = True

    # Collect input files corresponding to each output file
    if conf["link_existing_feature_files"]:
        linkable_files_tss = TimestepStringFormatter(
            conf["existing_feature_file_fmt"]
        ).run(kwas["conf_in"]["timesteps"])
    else:
        linkable_files_tss = infiles_tss
    linkable_files_tss = sorted(linkable_files_tss.items())
    linked_files_tss = {}
    for tss_out, outfile in sorted(outfiles_tss.items()):
        linked_files_tss[outfile] = {}
        for tss_in, infile in copy(linkable_files_tss):
            if any(ts in tss_in for ts in tss_out):
                linked_files_tss[outfile][tss_in] = infile
                linkable_files_tss.remove((tss_in, infile))
    conf["linked_files_tss"] = linked_files_tss


def preproc_args__tracker(parser, kwas, conf):

    conf["reshrink_tracked_features"] = kwas["conf_idfy"]["grow_features_n"] > 0

    alpha = conf.pop("alpha")
    if alpha < 0 or alpha > 1:
        err = "Invalid alpha, must be in range (0.0, 1.0): {}".format(alpha)
        raise ValueError(err)
    conf["f_overlap"] = alpha
    conf["f_size"] = 1 - alpha

    sts = str(kwas["conf_in"]["timesteps"][0])
    if len(sts) == 8:
        conf["ts_fmt"] = "%Y%m%d"
    elif len(sts) == 10:
        conf["ts_fmt"] = "%Y%m%d%H"
    elif len(sts) == 12:
        conf["ts_fmt"] = "%Y%m%d%H%M"
    else:
        parser.error("cannot deduce timestep format: {}".format(sts))


def preproc_args__track_pp(parser, kwas, conf):
    pass


def preproc_args__exe(parser, kwas, conf):

    if conf["timings_logfile"] is not None:
        conf["timings_measure"] = True

    if conf["profile_nlines"] is None:
        conf["profile"] = False
    else:
        conf["profile"] = True
        if not conf["profile_nlines"]:
            conf["profile_nlines"] = 20
        elif len(conf["profile_nlines"]) != 1:
            parser.error("pass either no or one number to --profile")
        else:
            conf["profile_nlines"] = next(iter(conf["profile_nlines"]))


def pre_main():
    parser = setup_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    # SR_DBG<
    pr = cProfile.Profile()
    # pr.enable()
    # SR_DBG>
    kwas = vars(parser.parse_args())
    print_args(kwas)
    preproc_args(parser, kwas)
    # SR_DBG<
    # pr.disable()
    # ps = pstats.Stats(pr)
    # nlines = 10
    # for psort in ["tottime", "cumtime"]:
    #     ps.strip_dirs().sort_stats(psort).print_stats(nlines)
    # exit(0)
    # SR_DBG>
    main(**kwas)


if __name__ == "__main__":
    pre_main()
