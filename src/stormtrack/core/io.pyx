# !/usr/bin/env python3

from __future__ import print_function

# C: Third-party
cimport cython
cimport numpy as np

# Standard library
import errno
import json
import logging as log
import os
import re
import sys
from collections import OrderedDict as odict
from copy import copy
from copy import deepcopy
from functools import total_ordering
from pprint import pprint

# Third-party
import cython
import h5py
import numpy as np
import PIL
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Local
from ..utils.netcdf import points_lonlat_to_inds
from ..utils.various import NoIndent
from ..utils.various import NoIndentEncoder
from .identification import Feature
from .identification import feature2d_from_jdat
from .tracking import FeatureTrack
from .tracking import remerge_partial_tracks


# SR_TMP < SR_TODO if retained, move into more appropriate module
# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef int mask_sum(np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int i, j, nx = mask.shape[0], ny = mask.shape[1]
    cdef int sum=0
    for i in range(nx):
        for j in range(ny):
            if mask[i, j]:
                sum += 1
    return sum
# SR_TMP >


# SR_TMP < SR_TODO if retained, move into more appropriate module
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=1] sum_hist_bins(
        np.ndarray[np.float32_t, ndim=2] fld,
        np.ndarray[np.float32_t, ndim=1] bins):
    cdef int i, j, k, nx=fld.shape[0], ny=fld.shape[0]
    cdef int nbins = len(bins) + 1
    cdef np.ndarray[np.float32_t, ndim=1] sums = np.zeros(nbins, np.float32)
    cdef np.float32_t v
    for i in range(nx):
        for j in range(ny):
            v = fld[i, j]
            for k in range(nbins):
                if v < bins[k]:
                    sums[k] += v
                    break
            else:
                sums[nbins] += v
    return sums
# SR_TMP >


# SR_TMP < SR_TODO if retained, move into more appropriate module
# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=1] count_hist_bins(
        np.ndarray[np.float32_t, ndim=2] fld,
        np.ndarray[np.float32_t, ndim=1] bins):
    cdef int i, j, k, nx=fld.shape[0], ny=fld.shape[0]
    cdef int nbins = len(bins)
    cdef np.ndarray[np.int32_t, ndim=1] counts = np.zeros(nbins + 1, np.int32)
    cdef np.float32_t v
    for i in range(nx):
        for j in range(ny):
            v = fld[i, j]
            for k in range(nbins):
                if v < bins[k]:
                    counts[k] += 1
                    break
            else:
                counts[nbins] += 1
    return counts
# SR_TMP >


def write_feature_file(
        outfile,
        *,
        nx, ny,
        feature_name,

        features          = None,
        tracks            = None,
        info_features     = None,
        info_tracks       = None,

        jdat_old          = None,

        pixelfile_format  = "h5",
        pixel_store_mode  = "pixels",
        track_store_mode  = "json",

        graphfile         = None,
        link_features     = False,
        feature_files_tss = None,

        silent            = False,
    ):
    """Write features and optionally tracks to disk.

    Parameters
    ----------

    TODO

    """
    # Prepare Arguments - Shared

    # Features v. tracks
    if features is not None and tracks is not None:
        raise ValueError("mutually exclusive arguments: features, tracks")
    if features is not None:
        if info_features is None:
            raise ValueError("must pass info_features alongside features")
    if tracks is not None:
        if info_features is None:
            raise ValueError("must pass info_features alongside tracks")
        if info_tracks is None:
            raise ValueError("must pass info_tracks alongside tracks")

    # Feature pixels file
    if pixelfile_format not in ["npz", "h5"]:
        raise ValueError("unknown pixelfile format: "+pixelfile_format)
    if outfile.endswith(".json"):
        outfile_pixels = outfile.replace(".json", "."+pixelfile_format)
    elif outfile.endswith(".pickle"):
        outfile_pixels = outfile.replace(".pickle", "."+pixelfile_format)
    else:
        raise ValueError("unknown outfile format: "+outfile)

    # Create directory
    try:
        os.makedirs(os.path.dirname(outfile))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Prepare Arguments - Tracks Only

    # Track graphs file
    if tracks is None:
        if graphfile is not None:
            raise ValueError("must pass tracks alongside graphfile")
    else:
        if graphfile is not None:
            link_graphs = True
            outfile_graphs = graphfile
        else:
            link_graphs = False
            if outfile.endswith(".json"):
                outfile_graphs = outfile.replace(".json", ".graphs.pickle")
            elif outfile.endswith(".pickle"):
                outfile_graphs = outfile.replace(".pickle", ".graphs.pickle")

    # Link features
    if tracks is None:
        if link_features:
            raise ValueError("can only link features if tracks are passed")
    else:
        if link_features and (feature_files_tss, jdat_old) == (None, None):
            raise ValueError("must pass feature_files or jdat_old "
                    "alongside link_features")
        # Feature files
        if isinstance(feature_files_tss, dict):
            feature_files_tss = [[tss, fs] for tss, fs
                            in sorted(feature_files_tss.items())]


    if not silent:
        if tracks is None:
            msg = "write {} {} features to {}".format(
                    len(features), feature_name, outfile)
        else:
            msg = "write {} {} tracks to {}".format(
                    len(tracks), feature_name, outfile)
        log.info(msg)

    jdat = odict()

    # Header

    # Initialize header
    jdat["header"] = odict([("nx", nx), ("ny", ny)])

    if tracks is None:
        jdat["header"]["n_features"] = len(features)
    else:
        # Collect summary/meta data
        n_tracks = len(tracks)
        n_tracks_part = len([t for t in tracks if not t.is_complete()])
        n_features = sum([t.n for t in tracks])
        jdat["header"]["n_tracks"]         = n_tracks
        jdat["header"]["n_partial_tracks"] = n_tracks_part
        jdat["header"]["n_features"]       = n_features

    # Additional output file(s)
    jdat["header"]["pixelfile"] = os.path.basename(outfile_pixels)
    if tracks is not None:
        jdat["header"]["graphfile"] = os.path.basename(outfile_graphs)

    # Timesteps
    if tracks is None:
        timesteps = sorted(set([f.timestep for f in features]))
    else:
        timesteps = sorted(set([d for t in tracks for d in t.get_timesteps()]))
    if outfile.endswith(".json"):
        jdat["header"]["timesteps"] = NoIndent(timesteps)
    else:
        jdat["header"]["timesteps"] = timesteps

    if tracks is not None:
        # Whether and from where to link features
        jdat["header"]["link_features"]     = link_features
        jdat["header"]["feature_files_tss"] = feature_files_tss

    # Feature (and Track) Info

    # Add pixel store mode to info blocks
    info_features["pixel_store_mode"] = pixel_store_mode

    # Add features info
    jdat["info_"+feature_name] = info_features

    if tracks is not None:
        # Add tracks info
        jdat["info_tracks"] = info_tracks

    # SR_TMP < TODO properly integrate graph-based track storage
    if tracks is not None:
        if track_store_mode == "json":
            pass
        elif track_store_mode == "graph":
            # SR_TMP <
            separate_pixels_file = True
            store_feature_values = (pixel_store_mode == "pixels")
            # SR_TMP >
            __tmp__write_tracks_features_as_graphs(
                    outfile, tracks, feature_name,
                    separate_pixels_file    = separate_pixels_file,
                    store_values            = store_feature_values,
                    header                  = jdat["header"],
                    info_tracks             = jdat["info_tracks"],
                    info_features           = jdat["info_"+feature_name],
                )
            # +++++
            return
            # +++++
        else:
            err = "invalid track store mode '{}'".format(track_store_mode)
            raise ValueError(err)
    # SR_TMP >
    # Feature (and Track) Data

    # Add features data
    if tracks is None:
        key = "features_"+feature_name
        jdat[key] = []
        for feature in features:
            jdat_feature = feature.json_dict()
            jdat[key].append(jdat_feature)
    else:
        if not link_features:
            # Collect features and their data
            features = [f for t in tracks for f in t.features()]
            _r = _collect_jdat_features(features, timesteps, pixel_store_mode)
            jdat["features_"+feature_name] = _r

    if tracks is not None:
        # Add tracks data
        jdat["tracks"] = []
        for track in tracks:
            jdat_track = track.json_dict()
            jdat["tracks"].append(jdat_track)


    if jdat_old:
        # Merge in old jdat dict
        # Add what's missing, don't overwrite what's there
        for section, content in jdat_old.items():
            # SR_TMP <
            if section == "features" and not content:
                continue
            # SR_TMP >
            if section not in jdat:
                jdat[section] = content
            elif (section in ["header", "info"] or
                    section.startswith("info_")):
                for key, val in content.items():
                    if key not in jdat[section]:
                        jdat[section][key] = val

    if link_features:
        # Add features files to header of old JDAT (only for tracks)
        if feature_files_tss is None:
            # Collect (and remove) feature files from old jdats
            feature_files_tss = odict()
            feature_files_tss_old = odict([
                    ((tuple(k) if isinstance(k, list) else k), v)
                            for k, v in jdat_old["header"].pop(
                                    "feature_files_tss")])
            for tss, feature_file in feature_files_tss_old.items():
                if tuple(tss) in feature_files_tss:
                    if feature_file != feature_files_tss[tss]:
                        err = ("conflict: different feature files found "
                                "for {} timesteps [{}..{}]: {} != {}"
                                ).format(min(tss), max(tss), feature_file,
                                feature_files_tss[tss])
                        raise Exception(err)
                feature_files_tss[tuple(tss)] = feature_file
        else:
            if jdat_old is not None:
                # Remove feature files from old jdat
                del jdat_old["header"]["feature_files_tss"]


    # Write json data to disk
    if outfile.endswith(".json"):
        # Dump json string
        jstr = json.dumps(jdat, indent=4, cls=NoIndentEncoder)

        # Write json string to file
        with open(outfile, "w") as fo:
            fo.write(jstr)

    elif outfile.endswith(".pickle"):
        # Dump to pickle file
        with open(outfile, "wb") as fo:
            pickle.dump(jdat, fo, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise ValueError("unknown outfile format: "+outfile)

    if not link_features:
        # Write features pixels to NPZ or H5 file
        write_feature_pixels(
                outfile_pixels,
                feature_name = feature_name,
                features     = features,
                mode         = pixel_store_mode,
                silent       = silent,
            )

    if tracks is not None:

        # Unlink features; feature ids for each vertex are kept
        for track in tracks:
            track.unlink_features()

        if not link_graphs:
            # Save graphs to pickle file
            graphs_table = {t.id: t.graph for t in tracks}
            if not silent:
                log.info("save {} graphs to {}".format(
                        len(graphs_table), outfile_graphs))
            with open(outfile_graphs, "wb") as fo:
                pickle.dump(graphs_table, fo)


def write_feature_pixels(outfile, *, feature_name, features, mode,
        silent=False):
    """Write all feature pixels to an npz archive (shells/holes separately).

    Either, all pixels are writte to file ('pixels' mode), or only the shells
    and holes pixels ('boundaries' mode). In the latter case, the feature
    pixels are reconstructed from the boundaries. Note that pixel values are
    only written in pixels mode but lost in boundaries mode.
    """
    if mode not in ("pixels", "boundaries"):
        err = "invalid mode {}".format(mode)
        raise ValueError(err)
    if not silent:
        log.info("write pixel data ({}) to {}".format(mode, outfile))
    tables = {}
    for feature in features:

        # Store all pixels
        if mode == "pixels":
            key_pixels = "{}_{}_pixels".format(feature_name, feature.id)
            key_values = "{}_{}_values".format(feature_name, feature.id)
            tables[key_pixels] = feature.pixels
            tables[key_values] = feature.values

        # Store shells pixels
        for i, shell in enumerate(feature.shells):
            key_shell = "{}_{}_shell_{}".format(feature_name, feature.id, i)
            tables[key_shell] = shell

        # Store holes pixels
        if feature.holes is not None:
            for i, hole in enumerate(feature.holes):
                key_hole = "{}_{}_hole_{}".format(feature_name, feature.id, i)
                tables[key_hole] = hole

    if outfile.endswith(".npz"):
        np.savez_compressed(outfile, **tables)
    elif outfile.endswith(".h5"):
        with h5py.File(outfile, "w") as fo:
            for key, dat in tables.items():
                fo.create_dataset(key, data=dat)
    else:
        err = "unknown file format: {}".format(outfile)
        raise Exception(err)


def _collect_jdat_features(features, timesteps, pixel_store_mode):
    """Collect data of features and add it to json data dict."""

    jdat_features = []
    for feature in features:
        jdat_feature = feature.json_dict()

        if pixel_store_mode == "boundaries":
            # SR_DBG <
            nold = feature.properties.pop("n", feature.n)
            if nold != feature.n:
                # SR_TODO Debug this issue!!!!
                # Problem with "boundaries" mode: some features
                # mysteriously gain pixels (though not all)! First,
                # when this happened to all pixels, the problem was
                # that the holes were not correctly subtracted from
                # the feature masks, but this now generally works...
                err = ("number of pixels of {} changed: {} -> {}"
                        ).format(feature.id, nold, feature.n)
                raise Exception(err)
                log.warning(err)
            # SR_DBG >

            stats = dict(
                    min    = feature.properties.pop("min",    -1),
                    mean   = feature.properties.pop("mean",   -1),
                    median = feature.properties.pop("median", -1),
                    max    = feature.properties.pop("max",    -1),
                )
            jdat_feature["stats"].update(stats)

        jdat_features.append(jdat_feature)

    return jdat_features


def __tmp__write_tracks_features_as_graphs(outfile, tracks, feature_name, *,
        header, info_tracks, info_features,
        separate_pixels_file=True, store_values=False, silent=False
    ):

    # Prepare output file name(s)
    if not outfile.endswith(".pickle"):
        raise ValueError("wrong suffix of graphs file: "+outfile)
    if separate_pixels_file:
        outfile_pixels = outfile.replace(".pickle", ".pixels.h5")

    # Reduce tracks to graphs
    _r = tracks_to_graphs(tracks, separate_pixels=separate_pixels_file,
            store_values=store_values)
    graphs_by_tid = _r["graphs_by_tid"]
    if separate_pixels_file:
        pixel_data_by_tid = _r["pixel_data_by_tid"]

    # Update header
    header["store_feature_values"] = store_values
    header["separate_pixels_file"] = separate_pixels_file
    if separate_pixels_file:
        header["pixels_file"] = outfile_pixels

    # Collect output data
    data_out = dict(
            header        = header,
            info_tracks   = info_tracks,
            info_features = info_features,
            graphs_by_tid = graphs_by_tid,
        )

    # Write tracks to disk as graphs
    if not silent:
        print("write tracks and features as graphs to "+outfile)
    with open(outfile, "wb") as fo:
        pickle.dump(data_out, fo)

    if separate_pixels_file:
        # Write pixels data to disk
        if not silent:
            print("write pixels data to "+outfile_pixels)
        with h5py.File(outfile_pixels, "w") as fo:

            for tid, track_data in sorted(pixel_data_by_tid.items()):
                grp_tid = fo.create_group(str(tid))

                for fid, shells in sorted(track_data["shells_by_fid"].items()):
                    grp_fid = grp_tid.create_group(str(fid))

                    for i, shell in enumerate(shells):
                        key_shell = "shell__{}".format(i)
                        grp_fid.create_dataset(key_shell, data=shell)

                    holes = track_data["holes_by_fid"][fid]
                    for i, hole in enumerate(holes):
                        key_hole = "hole__{}".format(i)
                        grp_fid.create_dataset(key_hole, data=hole)

                    if store_values:
                        values = track_data["values_by_fid"][fid]
                        grp_fid.create_dataset("values", data=values)

def tracks_to_graphs(tracks, *, separate_pixels=False, store_values=False):
    """Reduce tracks to graphs with data to rebuild tracks and features."""

    graphs_by_tid = odict()
    if separate_pixels:
        pixel_data_by_tid = {}
    for track in tracks:
        _r = track_to_graph(track, separate_pixels=separate_pixels,
                store_values=store_values)
        graphs_by_tid[track.id] = _r["graph"]
        if separate_pixels:
            pixel_data_by_tid[track.id] = {}
            for key_base in ["shells", "holes", "values"]:
                key = key_base+"_by_fid"
                if key in _r:
                    pixel_data_by_tid[track.id][key] = _r[key]

    output = dict(graphs_by_tid=graphs_by_tid)
    if separate_pixels:
        output["pixel_data_by_tid"] = pixel_data_by_tid
    return output

def track_to_graph(track, *, separate_pixels=False, store_values=False):
    """Reduce track to graph with data to rebuild track and features.

    The information necessary to rebuild the track and feature objects later
    is included as graph (track) and vertex (features) attributes.

    Parameters
    ----------

    track : Track
        Track object, containing graph and feature objects.

    separate_pixels : bool, optional (default: False)
        Return pixel data (shell, holes, optionally pixel values)
        separately instead of storing it in the graph as vertex attributes.

    store_values : bool, optional (default: False)
        Store pixel values alongside shell and holes (memory-intensive).
    """
    # Retain original graph
    graph = copy(track.graph)

    # Store feature data as vertex attributes
    if separate_pixels:
        shells_by_fid = {}
        holes_by_fid = {}
        if store_values:
            values_by_fid = {}
    for vx in graph.vs:
        feature = vx["feature"]
        vx["feature_id"] = feature.id
        vx["feature_json_dict"] = feature.json_dict(format=False)
        if separate_pixels:
            shells_by_fid[feature.id] = feature.shells
            holes_by_fid[feature.id] = feature.holes
            if store_values:
                values_by_fid[feature.id] = feature.values
        else:
            vx["feature_shells"] = feature.shells
            vx["feature_holes"] = feature.holes
            if store_values:
                vx["feature_values"] = feature.values

    # Remove feature objects
    del graph.vs["feature"]

    # Store track data as graph attributes
    graph["track_json_dict"] = track.json_dict(format=False)

    output = dict(graph=graph)
    if separate_pixels:
        output["shells_by_fid"] = shells_by_fid
        output["holes_by_fid"] = holes_by_fid
        if store_values:
            output["values_by_fid"] = values_by_fid
    return output


def distribute_tracks_across_outfiles(tracks, timesteps_outfiles, jdats,
        args_methods=None, group_names=None, features_have_pixels=True,
        silent=False):
    """Distribute tracks across outfiles and split them at borders.

    Requires tracks and a list of tuples: ((ts_min, ts_max), outfile).

    Optionally, tracks are sorted into two groups based thresholds applied
    to one or more track methods, for which arguments can be passed as well.
    For example, to split by total duration at 24 h, call with:

        args_methods = {
                "name"      : "duration",
                "kwas"      : {"total": True},
                "threshold" : 24,
            }

    Note that outfile names must contain '{GROUP}' where the group names,
    which must be passed alongside args_methods, are inserted.

    Returns a dict with outfiles as keys and corresponding tracks as values.
    """
    key_info = "info_tracks"

    n = len(timesteps_outfiles)
    if n > 1:
        if not silent:
            log.info("split tracks for {} outfiles".format(n))

    if len(jdats) != len(timesteps_outfiles):
        err = "must pass as many jdats as outfiles"
        raise ValueError(err)

    # Simple case without sorting tracks by method value
    if not args_methods:
        outfiles_tracks = {}
        for _arg, jdat in zip(timesteps_outfiles, jdats):
            if len(_arg) == 2:
                (ts_min, ts_max), outfile = _arg
                skip = False
            elif len(_arg) == 3:
                (ts_min, ts_max), outfile, skip = _arg

            if not silent:
                log.info(" - ({}..{}) {}".format(ts_min, ts_max, outfile))
            tracks_outfile = []
            for track in tracks.copy():
                if track.ts_start(total=False) > ts_max:
                    continue
                elif track.ts_end(total=False) <= ts_max:
                    tracks.remove(track)
                    partial_track = track
                else:
                    # if not silent:
                    #    log.info("   -> split track {}".format(track.id))
                    partial_track = track.cut_off(until=ts_max,
                            compute_footprint=features_have_pixels)
                tracks_outfile.append(partial_track)
            outfiles_tracks[outfile] = (jdat, tracks_outfile)
        return outfiles_tracks

    # SR_TODO Merge simple (above) and extended case (below)

    # Extended case where tracks are sorted into two groups by method value
    if group_names is None:
        err = "must pass group_names alongside args_methods"
        raise ValueError(err)
    if len(group_names) != 2:
        err = "group_names must be omitted or contain exactly two elements"
        raise ValueError(err)

    # Collect group info for track info
    group_name_lt, group_name_ge = group_names
    group_args_lt = odict(sorted([(g["name"],
            ("lt", g["threshold"])) for g in args_methods]))
    group_args_ge = odict(sorted([(g["name"],
            ("ge", g["threshold"])) for g in args_methods]))
    group_info_lt = odict([
            ("name", group_name_lt),
            ("args", group_args_lt),
        ])
    group_info_ge = odict([
            ("name", group_name_ge),
            ("args", group_args_ge),
        ])

    outfiles_tracks = {}
    for _arg, jdat in zip(timesteps_outfiles, jdats):
        if len(_arg) == 2:
            (ts_min, ts_max), outfile = _arg
            skip = False
        elif len(_arg) == 3:
            (ts_min, ts_max), outfile, skip = _arg
        if not silent:
            log.info(" - ({}..{}) {}".format(ts_min, ts_max, outfile))

        if "{GROUP}" not in outfile:
            # Locate timestep in file name and insert {GROUP} before it
            # (This is a pragmatic solution that fits my current use case)
            if outfile.endswith(".json"):
                suffix = "json"
            elif outfile.endswith(".pickle"):
                suffix = "pickle"
            else:
                raise ValueError("unknown outfile format: "+outfile)
            sig_ts = (r'[12][0-9][0-9][0-9]'
                    r'(?P<mmddhh>[01][0-9]'
                    r'(?P<ddhh>[0-3][0-9]'
                    r'(?P<hh>[0-2][0-9])?)?)?')
            rx_str = r'(?P<pre>.*)_(?P<ts>' + sig_ts + r').'+suffix
            match = re.match(rx_str, outfile)
            outfile = "{}_{}_{}.{}".format(match.group("pre"), "{GROUP}",
                    match.group("ts"), suffix)

        # Add group names to outfiles
        outfile_lt = outfile.format(GROUP=group_name_lt)
        outfile_ge = outfile.format(GROUP=group_name_ge)

        # Add group info to track info
        jdat_lt = deepcopy(jdat)
        jdat_ge = deepcopy(jdat)
        jdat_lt[key_info]["grouping"] = group_info_lt
        jdat_ge[key_info]["grouping"] = group_info_ge

        # Assign, split, and sort tracks
        tracks_outfile_lt = []
        tracks_outfile_ge = []
        for track in tracks.copy():

            # Check timestep range
            if track.ts_start(total=False) > ts_max:
                continue

            # Check methods values
            # Note: Currently, a track is only sorted into the "ge" group
            # if it is at/above the thresholds of all methods; if it's
            # below even one threshold, it's sorted into the "lt" group
            # TODO more flexible implementation
            for args_method in args_methods:
                try:
                    method = getattr(track, args_method["name"])
                except AttributeError:
                    err = "invalid method_name for class {}: {}".format(
                            track.__class__.__name__, args_method["name"])
                    raise ValueError(err)
                val = method(**args_method["kwas"])
                if val < args_method["threshold"]:
                    dest = tracks_outfile_lt
                    break
            else:
                dest = tracks_outfile_ge

            # Split track (if necessary)
            if track.ts_end(total=False) <= ts_max:
                tracks.remove(track)
                partial_track = track
            else:
                # print("   -> split track {}".format(track.id))
                partial_track = track.cut_off(until=ts_max,
                        compute_footprint=features_have_pixels)

            # Store partial track (other will be further processed)
            dest.append(partial_track)

        if not skip:
            outfiles_tracks[outfile_lt] = (jdat_lt, tracks_outfile_lt)
            outfiles_tracks[outfile_ge] = (jdat_ge, tracks_outfile_ge)

    return outfiles_tracks


def read_feature_files(infiles, *, feature_name, silent=False, silent_core=None,
        counter=False, counter_core=False, extra_tracks=None,
        is_subperiod=True, optimize_pixelfile_input=False, **kwas):
    """Read multiple feature files; remerge partial tracks if necessary.

    See 'read_feature_file' for a detailed parameter description.

    """
    if silent_core is None:
        silent_core = silent
    if counter_core is None:
        counter_core = counter

    # SR_TMP < TODO only define defaults in one single place!
    read_pixels                 = kwas.get("read_pixels", True)
    read_tracks                 = kwas.get("read_tracks", True)
    read_pixelfile              = kwas.get("read_pixelfile", True)
    rebuild_pixels              = kwas.get("rebuild_pixels", False)
    rebuild_pixels_if_necessary = kwas.get("rebuild_pixels_if_necessary", False)
    retain_n_biggest_tracks     = kwas.get("retain_n_biggest_tracks", None)
    discard_untracked_features  = kwas.get("discard_untracked_features", False)
    # SR_TMP >

    # Initialize lists for features and tracks
    features = []
    if extra_tracks is None:
        tracks = []
    else:
        tracks = [t for t in extra_tracks]

    if optimize_pixelfile_input:
        kwas["read_pixelfile"] = False
        # SR_DBG <
        # +kwas["rebuild_pixels_if_necessary"] = False
        # SR_DBG >

    jdats = []
    nskip_features = 0
    nskip_tracks   = 0
    pixelfiles_fids = {}
    timesteps = []
    nfiles = len(infiles)
    for ifile, infile in enumerate(infiles):

        if not silent and not counter:
            log.info("reading {}".format(infile))

        if counter:
            pct = ifile/nfiles
            msg = " {: 2.0%} reading {}".format(pct, infile)
            try:
                w, h = os.get_terminal_size()
            except OSError:
                pass
            else:
                print(msg[:w], end="\r", flush=True)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++
        _r = read_feature_file(
                infile,
                feature_name             = feature_name,
                silent                   = silent,
                optimize_pixelfile_input = False,
                **kwas)
        new_tracks          = _r["tracks"]
        new_features        = _r["features"]
        new_pixelfiles_fids = _r["pixelfiles_fids"]
        new_timesteps       = _r["timesteps"]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++

        if not silent and not counter:
            log.info(" -> {:,} tracks and {:,} features".format(
                    len(new_tracks), len(new_features)))

        # Store new features, tracks, etc.
        features.extend(new_features)
        tracks.extend(new_tracks)
        jdats.append(_r["jdat"])
        if new_pixelfiles_fids is not None:
            pixelfiles_fids.update(new_pixelfiles_fids)
        if new_timesteps is not None:
            timesteps = sorted(set(timesteps + new_timesteps))

        nskip_features += _r["nskip_features"]
        nskip_tracks   += _r["nskip_tracks"]

        if read_tracks and retain_n_biggest_tracks:
            # Remove small tracks on-the-fly
            # Note: ts_start etc. already applied inside read_feature_file
            _r = select_tracks_features(
                    tracks                     = tracks,
                    features                   = features,
                    nskip_tracks               = nskip_tracks,
                    nskip_features             = nskip_features,
                    retain_n_biggest_tracks    = retain_n_biggest_tracks,
                    discard_untracked_features = discard_untracked_features,
                    silent                     = silent_core,
                )
            tracks          = _r["tracks"]
            features        = _r["features"]
            nskip_tracks    = _r["nskip_tracks"]
            nskip_features  = _r["nskip_features"]

    if counter:
        try:
            w, h = os.get_terminal_size()
        except OSError:
            pass
        else:
            print(" "*w, end="\r", flush=True)

    # Print number of files
    if not silent:
        nf_read = len(features)
        nf_tot = nf_read + nskip_features
        nt_read = len(tracks)
        nt_tot = nt_read + nskip_tracks
        nf_tracked = len([f for f in features if f.track() is not None])
        log.info(("read {:,}/{:,} features and {:,}/{:,} tracks containing "
                "{:,} features from {} files").format(nf_read, nf_tot,
                nt_read, nt_tot, nf_tracked, len(infiles)))

    if optimize_pixelfile_input and read_pixelfile:

        # SR_TMP <
        # Grab the next best jdat (only header used; should be all the same)
        jdat = next(iter(jdats))
        period = "{}..{}".format(min(timesteps), max(timesteps))
        # SR_TMP >

        # Read pixelfiles of the remaining features from disk
        _fids = [f.id for f in features]
        features_read_pixels(feature_name, features, pixelfiles_fids, jdat,
                read_pixels=read_pixels, rebuild_pixels=rebuild_pixels,
                rebuild_pixels_if_necessary=rebuild_pixels_if_necessary,
                fids_select=_fids, counter=True, period=period,
            )

    if read_tracks and len(infiles) > 1 or extra_tracks is not None:
        # Remerge partial tracks
        if not silent:
            log.info("remerge partial tracks among {:,}".format(len(tracks)))
        nold = len(tracks)
        tracks = remerge_partial_tracks(tracks, counter=counter_core,
                is_subperiod=is_subperiod)
        if not silent:
            log.info(" -> re-built {:,} tracks from {:,}".format(len(tracks), nold))

    return dict(
            features        = features,
            tracks          = tracks,
            jdats           = jdats,
            nskip_features  = nskip_features,
            pixelfiles_fids = pixelfiles_fids,
            timesteps       = timesteps,
        )

# SR_TODO Implement named tracks analogous to named features!
# SR_TODO (currently features are returned in name dict, tracks in list)
def read_feature_file(
        infile,
        *,
        feature_name,
        infile_type                    = "json",

        read_pixelfile                 = True,
        read_pixels                    = True,
        rebuild_pixels                 = False,
        rebuild_pixels_if_necessary    = False,

        graph_file_format              = "basic",

        minsize                        = 1,
        maxsize                        = -1,
        timesteps                      = None,
        ts_start                       = None,
        ts_end                         = None,
        tss_excl_mode_tracks           = "liberal",
        tss_excl_mode_tracked_features = "liberal",

        read_tracks                    = True,
        track_store_mode               = "json",
        discard_untracked_features     = False,
        retain_n_biggest_tracks        = None,
        optimize_pixelfile_input       = False,
        timestep                       = None,

        silent                         = False,
        counter                        = False,

        # SR_TMP < TODO resolve underlying issues
        ignore_missing_neighbors              = True,
        ignore_edges_pshare_0                 = True,
        ignore_missing_total_track_stats      = False,
        ignore_missing_missing_features_stats = False,
        # SR_TMP >
    ):
    """Read a '.json' file containing features and/or tracks, and linked files.

    Parameters
    ----------

    infile : str
        Input file containing features/tracks (JSON, PICKLE).

    feature_name : str
        Name of features to be read.

    infile_type : str, optional (default: 'json')
        Type of the input file. Options:

         - 'json': classic format with feature and -- optionally -- track
                data as JSON dict; either JSON or PICKLE file; accompanied
                by pixle (NPZ or H5) and -- optionally -- track graph (igraph)
                file (PICKLE);

         - 'graph': compact format with all track and feature info stored
                in track graph (igraph) file (PICKLE).

    read_pixelfile : bool, optional (default: True)
        Feature pixels, shells pixels, etc. from the respective
        NPZ or H5 file.

    read_pixels : bool, optional (default: True)
        Read feature pixels from the respective NPZ or H5 file;
        note that shells pixels etc. are read either way.

    rebuild_pixels : bool, optional (default: False)
        Rebuild feature pixels from the respective shells and holes
        pixels, in cases where the pixels are absent or not read.

    rebuild_pixels_if_necessary : bool, optional (default: False)
        Rebuild pixels if features are stored in boundary mode.

    graph_file_format: TODO

    minsize : int, optional (default: 1)
        Minimum feature size; set to < 2 to disable.

    maxsize : int, optional (default: -1)
        Maximum feature size; set to < 1 to disable.

    ts_start : int, optional (default: None)
        Start timestep of an optional target period; features and tracks
        partially or fully outside the target period may be discarded,
        depending on 'tss_excl_mode_tracks'.

    ts_end : int, optional (default: None)
        End timestep of an optional target period; features and tracks
        partially or fully outside the target period may be discarded,
        depending on 'tss_excl_mode_tracks'.

    timesteps : list of int, optional (default: None)
        Only read features at these timesteps. TODO solution for tracks

    tss_excl_mode_tracks : str, optional (default: "liberal")
        How to apply the timestep thresholds 'ts_start' and/or 'ts_end', if
        given. Options:

         - "strict"
              Exclude all tracks that start before ts_start and/or end after
              ts_end, i.e., also those which lie partially outside the target
              period; and all features before ts_start and/or after ts_start.

         - "liberal"
              Only exclude those tracks which end before ts_start or start
              after ts_end, i.e., which lie fully outside the target period;
              features are always first read (potentially slow) and those
              outside the target period that don't belong to a retained
              track are removed.

    tss_excl_mode_tracked_features : str, optional (default: "liberal")
        How to apply the timestep thresholds 'ts_start' and/or 'ts_end', if
        given, in concert with tss_excl_mode_tracks="liberal". Options:

        - "strict"
            Exclude all features outside the range (ts_start..ts_end),
            even if they belong to a track which is partially outside the
            range and thus retained. Note that the respective tracks will
            be only partially useable; this option should thus be used with
            care.

         - "liberal"
            Only exclude features outside the range (ts_start..ts_end) which
            are untracked, or belong to tracks fully outside the range;
            features outside the range which belong to retained tracks will
            also be retained.

    read_tracks : bool, optional (default: True)
        Read -- or omit -- tracks.

    track_store_mode : bool, optional (default: 'json')
        How tracks and their features are stored. Options:

        - "json"
            The track and feature object data is stored as a JSON dict,
            either in a JSON text file, or as a PICKLE dump.

        - "graph"
            The track and feature object data is stored directly in the
            track graphs as graph an vertex attributes.

    discard_untracked_features : bool, optional (default: False)
        Discard features that don't belong to any track.

    retain_n_biggest_tracks : int, optional (default: None)
        If set to a positive integer, only the respective number of tracks,
        sorted reversely by size, are returned, or less if there are not
        as many.

    optimize_pixelfile_input : bool, optional (default: False)
        Optimize feature input by first importing the features in
        absence of any pixels (including shells etc.), and only adding the
        pixels etc. in the end once many features have been eliminated;
        this is an optimization for extraordinarily heavy feature files in
        cases where many features are discarded anyway, e.g., because only
        a small number of tracks is selected based on a target period or
        the selection of N biggest tracks.

    timestep : int, optional (default: None)
        Timestep to which features are set, in cases where timesteps are
        missing in the input file; if necessary, something is likely amiss
        in some other placce.

    silent : bool, optional (default: False)
        Suppress most informational prints.

    counter : bool, optional (default: False)
        Show a counter with the current progress during input,
        which is constantly overwritten by itself or other prints, such that
        it doesn't clutter up the standard output; particularly useful in
        parallelized scripts.

    ignore_missing_neighbors : bool, optional (default: True)
        Suppress a warning related to missing neighbors;
        obsolete once the respective issue in the identification fixed.

    ignore_edges_pshare_0 : bool, optional (default: True)
        ignore edges with p_share equal 0.0.

    ignore_missing_total_track_stats : bool, optional (default: False)
        Ignore missing 'total track stats' of tracks, which are
        some stats about full tracks that make partial tracks usuable in
        some circumstances without reconstructing the full track from
        potentially many subtracks spread over as many files;
        obsolete once no more legacy files without such stats are around.

    ignore_missing_missing_features_stats : bool, optional (default: False)
        Ignore missing 'features stats' of tracks;
        obsolete once no more legacy files without such stats are around.

    """
    debug = False
    if debug: log.debug("< read_feature_file {}".format(infile))

    # SR_TMP < TODO integrate in existing function
    if track_store_mode == "graph":
        _r = __tmp__rebuild_tracks_from_graphs(
                infile,
                feature_name                     = feature_name,

                read_pixelfile                   = read_pixelfile,
                read_pixels                      = read_pixels,
                rebuild_pixels                   = rebuild_pixels,
                rebuild_pixels_if_necessary      = rebuild_pixels_if_necessary,

                graph_file_format                = graph_file_format,

                minsize                          = minsize,
                maxsize                          = maxsize,
                timesteps                        = timesteps,
                ts_start                         = ts_start,
                ts_end                           = ts_end,
                tss_excl_mode_tracks             = tss_excl_mode_tracks,
                tss_excl_mode_tracked_features   = tss_excl_mode_tracked_features,

                read_tracks                      = read_tracks,
                retain_n_biggest_tracks          = retain_n_biggest_tracks,
                optimize_pixelfile_input         = optimize_pixelfile_input,
                timestep                         = timestep,

                silent                           = silent,
                counter                          = counter,

                # SR_TMP <
                ignore_missing_neighbors              = ignore_missing_neighbors,
                ignore_edges_pshare_0                 = ignore_edges_pshare_0,
                ignore_missing_total_track_stats      = ignore_missing_total_track_stats,
                ignore_missing_missing_features_stats = ignore_missing_missing_features_stats,
                # SR_TMP >
            )
        tracks         = _r["tracks"]
        features       = _r["features"]
        nskip_tracks   = _r["nskip_tracks"]
        nskip_features = _r["nskip_features"]
        info_tracks    = _r["info_tracks"]
        info_features  = _r["info_features"]
        header         = _r["header"]

        # SR_TMP <
        jdat = {
                "header"                : header,
                "info_"+feature_name    : info_features,
                "info_tracks"           : info_tracks,
            }
        pixelfiles_fids = {}
        # SR_TMP >

    elif track_store_mode == "json":

        if timesteps is not None:
            if (ts_start, ts_end) != (None, None):
                raise ValueError("incompatible: timesteps, ts_start/ts_end")

        # SR_TMP <
        if infile_type != "json":
            raise NotImplementedError("infile type '{}'".format(infile_type))
        # SR_TMP >

        # Check and prepare tss_excl_mode_* arguments
        if tss_excl_mode_tracks not in ["strict", "liberal"]:
            err = "invalid tss_excl_mode_tracks: {}".format(tss_excl_mode_tracks)
            raise ValueError(err)
        if tss_excl_mode_tracked_features not in ["strict", "liberal"]:
            err = "invalid tss_excl_mode_tracked_features: {}".format(
                    tss_excl_mode_tracked_features)
            raise ValueError(err)
        if (tss_excl_mode_tracks == "strict" or
                tss_excl_mode_tracked_features == "strict"):
            ts_start_features = ts_start
            ts_end_features = ts_end
        else:
            ts_start_features = None
            ts_end_features = None

        if counter:
            msg = ".. loading {}".format(infile)
            try:
                w, h = os.get_terminal_size()
            except OSError:
                pass
            else:
                print(msg[:w], end="\r", flush=True)

        # Read JSON data (from text of binary file)
        if infile.endswith(".json"):
            # Read json file
            json_load = json.load
            with open(infile, "r") as fi:
                jdat = json_load(fi)
        elif infile.endswith(".pickle"):
            # Read pickle file
            with open(infile, "rb") as fi:
                jdat = pickle.load(fi)
            # SR_TMP <
            jdat_remove_noindent(jdat)
            # SR_TMP >
        else:
            raise ValueError("unknown infile format: "+infile)

        if "tracks" not in jdat.keys():
            read_tracks = False
            pre_read_tracks = False
        else:
            # SR_TMP < TODO figure out if this is still not supported
            if timesteps is not None and read_tracks:
                raise NotImplementedError("timesteps & read_tracks")
            # SR_TMP >
            pre_read_tracks = (read_tracks or
                    (ts_start, ts_end, timesteps) != (None, None, None))

        if counter:
            try:
                w, h = os.get_terminal_size()
            except OSError:
                pass
            else:
                print(" "*w, end="\r", flush=True)

        # Determine input file directory (for other files)
        indir = os.path.dirname(infile)
        if indir == "":
            indir = "."

        if jdat["header"]["n_features"] == 0:
            # Special case: no features
            return dict(
                    features        = [],
                    tracks          = [],
                    jdat            = jdat,
                    nskip_features  = 0,
                    nskip_tracks    = 0,
                    pixelfiles_fids = {},
                    timesteps       = [],
                )


        if not pre_read_tracks:
            tids_skip = set()
            fids_skip = set()
        else:
            # Read track graphs
            # Done here to determine which features to skip

            # Read graphs
            # SR_TMP <
            graphfile_header = os.path.basename(jdat["header"]["graphfile"])
            graphfile_deriv = "{}.graphs{}".format(
                    os.path.basename(os.path.splitext(infile)[0]),
                    os.path.splitext(graphfile_header)[1])
            if graphfile_header != graphfile_deriv:
                msg = ("warning: graphfile derived from infile differs from "
                        "that in header: {} != {}; reading the former"
                        ).format(graphfile_header, graphfile_deriv)
                print(msg)
            graphfile = "{}/{}".format(indir, graphfile_deriv)
            # SR_TMP >
            if debug: log.debug("read graphs from {}".format(graphfile))
            _r = read_track_graphs(graphfile,
                    ts_start    = ts_start,
                    ts_end      = ts_end,
                    timesteps   = timesteps,
                    silent      = silent,
                    tss_excl_mode_tracks = tss_excl_mode_tracks,
                    tss_excl_mode_tracked_features = tss_excl_mode_tracked_features,
                )
            graphs_tid = _r["graphs_by_tid"]
            tids_skip  = _r["tids_skip"]
            fids_skip  = _r["fids_skip"]

        # Import features
        if jdat["header"].get("link_features"):
            # Read features from linked files
            _feature_files_tss = odict([(tuple(tss), fs)
                    for tss, fs in jdat["header"]["feature_files_tss"]])
            _feature_files = _feature_files_tss.values()
            read_pixelfile_now = read_pixelfile
            if optimize_pixelfile_input:
                read_pixelfile_now = False
            # SR_TMP <
            if len(fids_skip) > 0:
                raise NotImplementedError("fids_skip for linked features")
            # SR_TMP >
            _r =  read_feature_files(
                    _feature_files,
                    names                       = feature_name,
                    read_pixelfile              = read_pixelfile_now,
                    read_pixels                 = read_pixels,
                    rebuild_pixels              = rebuild_pixels,
                    rebuild_pixels_if_necessary = rebuild_pixels_if_necessary,
                    minsize                     = minsize,
                    maxsize                     = maxsize,
                    ts_start                    = ts_start_features,
                    ts_end                      = ts_end_features,
                    tss_excl_mode_tracks        = tss_excl_mode_tracks,
                    read_tracks                 = False,
                    discard_untracked_features  = False,
                    timestep                    = timestep,
                    silent                      = True,
                    counter                     = counter,
                    ignore_missing_neighbors              = ignore_missing_neighbors,
                    ignore_missing_total_track_stats      = ignore_missing_total_track_stats,
                    ignore_missing_missing_features_stats = ignore_missing_missing_features_stats,
                )
            features        = _r["features"]
            nskip_features  = _r["nskip_features"]
            pixelfiles_fids = _r["pixelfiles_fids"]
            timesteps       = _r["timesteps"]
        else:
            # Read features from current files
            # SR_TMP <
            pixelfile_header = os.path.basename(jdat["header"]["pixelfile"])
            pixelfile_deriv = "{}{}".format(
                    os.path.basename(os.path.splitext(infile)[0]),
                    os.path.splitext(pixelfile_header)[1])
            if pixelfile_header != pixelfile_deriv:
                msg = ("warning: pixelfile derived from infile differs from "
                        "that in header: {} != {}; reading the former"
                        ).format(pixelfile_header, pixelfile_deriv)
                print(msg)
            pixelfile = "{}/{}".format(indir, pixelfile_deriv)
            # SR_TMP >
            if read_pixelfile:
                pixelfile_now = pixelfile
            else:
                pixelfile_now = None
            _r = rebuild_features(
                    pixelfile                    = pixelfile_now,
                    jdat                         = jdat,
                    feature_name                 = feature_name,
                    indir                        = indir,
                    timestep                     = timestep,
                    read_pixels                  = read_pixels,
                    rebuild_pixels               = rebuild_pixels,
                    rebuild_pixels_if_necessary  = rebuild_pixels_if_necessary,
                    minsize                      = minsize,
                    maxsize                      = maxsize,
                    ts_start                     = ts_start_features,
                    ts_end                       = ts_end_features,
                    fids_skip                    = fids_skip,
                    ignore_missing_neighbors     = ignore_missing_neighbors,
                    counter                      = counter,
                    silent                       = silent,
                )
            features        = _r["features"]
            nskip_features  = _r["nskip_features"]
            timesteps       = _r["timesteps"]
            _fids = sorted([f.id for f in features])
            pixelfiles_fids = {pixelfile: _fids}

        if not read_tracks:
            tracks = []
            nskip_tracks = 0
        else:
            # Note: graphs already read above (before features)

            track_config = jdat["info_tracks"]
            features_by_id = {feature.id: feature for feature in features}

            # Rebuild tracks
            _r = rebuild_tracks(
                    jdat_tracks          = jdat["tracks"],
                    features_by_id       = features_by_id,
                    graphs_tid           = graphs_tid,
                    track_config         = track_config,
                    ts_start             = ts_start,
                    ts_end               = ts_end,
                    tss_excl_mode_tracks = tss_excl_mode_tracks,
                    tids_skip            = tids_skip,
                    fids_skip            = fids_skip,
                    debug                = debug,
                    counter              = counter,
                    ignore_missing_total_track_stats      = ignore_missing_total_track_stats,
                    ignore_missing_missing_features_stats = ignore_missing_missing_features_stats,
                )
            tracks = _r["tracks"]
            nskip_tracks = _r["nskip_tracks"]
    else:
        raise ValueError("invalid track store mode: "+track_store_mode)

    # Remove some tracks and/or features
    _r = select_tracks_features(
            tracks                      = tracks,
            features                    = features,
            nskip_tracks                = nskip_tracks,
            nskip_features              = nskip_features,
            ts_start                    = ts_start,
            ts_end                      = ts_end,
            tss_excl_mode_tracks        = tss_excl_mode_tracks,
            retain_n_biggest_tracks     = retain_n_biggest_tracks,
            discard_untracked_features  = discard_untracked_features,
            silent                      = silent,
        )
    tracks          = _r["tracks"]
    features        = _r["features"]
    nskip_tracks    = _r["nskip_tracks"]
    nskip_features  = _r["nskip_features"]

    if track_store_mode != "graph": # SR_TMP TODO implement this as well
        if optimize_pixelfile_input:
            # Read pixelfiles of the remaining features from disk
            period = "{}..{}".format(min(timesteps), max(timesteps))
            features_read_pixels(feature_name, features, pixelfiles_fids, jdat,
                    read_pixels=read_pixels, rebuild_pixels=rebuild_pixels,
                    rebuild_pixels_if_necessary=rebuild_pixels_if_necessary,
                    counter=True, period=period,
                )

    if not silent:
        # Print number of features that have been read
        _nf = len(features)
        msg = "read {:,} features".format(_nf)
        if nskip_features > 0:
            msg += "; skipped {:,}/{:,} features".format(nskip_features,
                    (_nf + nskip_features))
        if tracks:
            n_tracked = len([f for f in features if f.track() is not None])
            msg += "; {:,}/{:,} features in {:,} tracks".format(
                    n_tracked, len(features), len(tracks))
            if nskip_tracks > 0:
                msg += "; skipped {:,}/{:,} tracks".format(nskip_tracks,
                        (len(tracks) + nskip_tracks))
        msg += ": {:,} {:}".format(len(features), feature_name)
        log.info(msg)

    return dict(
            features        = features,
            tracks          = tracks,
            jdat            = jdat,
            nskip_features  = nskip_features,
            nskip_tracks    = nskip_tracks,
            pixelfiles_fids = pixelfiles_fids,
            timesteps       = timesteps,
        )

def read_track_graphs(graphfile, *, format="basic",
        ts_start=None, ts_end=None, timesteps=None,
        tss_excl_mode_tracks="liberal",
        tss_excl_mode_tracked_features="liberal",
        ignore_edges_pshare_0=False, silent=False,
    ):

    with open(graphfile, "br") as fi:
        data_in = pickle.load(fi)
    if format == "basic":
        graphs_by_tid = data_in
    elif format == "extended":
        graphs_by_tid = data_in["graphs_by_tid"]
        header        = data_in["header"]
        info_tracks   = data_in["info_tracks"]
        info_features = data_in["info_features"]
    else:
        err = "invalid graph format '{}'; must be among {}".format(format,
                ["basic", "extended"])
        raise ValueError(err)

    if timesteps is not None and not isinstance(timesteps, set):
        timesteps = set(timesteps)

    # SR_TMP < Fix p_share; TODO fix underlying issue & remove once fixed
    nzeroisol = 0
    nzerobranch = 0
    for graph in graphs_by_tid.values():
        for edge in graph.es:
            if edge["p_share"] == 0:
                nold = len(graph.vs[edge.target].predecessors())
                nnew = len(graph.vs[edge.source].successors())
                if nold == 1 and nnew == 1:
                    edge["p_share"] = 1.0
                    nzeroisol += 1
                else:
                    nzerobranch += 1
                #    print("warning: p_share = 0 (nold={}, nnew={})".format(
                #            nold, nnew))
    if nzeroisol + nzerobranch > 0 and not ignore_edges_pshare_0:
        print(("warning: {:,} edges with p_share == 0 "
                "({:,} isolated, {:,} in branchings)").format(
                nzeroisol + nzerobranch, nzeroisol, nzerobranch))
    # SR_TMP >

    tids_skip = set()
    fids_skip = set()
    # Determine tracks and features to skip based on timesteps range
    ntot_tracks, ntot_features = len(graphs_by_tid), 0
    for tid, graph in graphs_by_tid.items():
        tss_track = set(graph.vs["ts"])
        ts_start_track = min(tss_track)
        ts_end_track = max(tss_track)
        ntot_features += len(graph.vs)

        if (ts_start, ts_end, timesteps) == (None, None, None):
            # Nothing to skip
            continue

        elif (ts_start, ts_end) != (None, None):

            # Check if track is partially or fully outside timesteps range
            skip_track_strict = track_is_outside_timestep_range(
                    ts_start, ts_end, "strict",
                    ts_start_track=ts_start_track,
                    ts_end_track=ts_end_track,
                )

            # Check if track is fully outside timesteps range
            skip_track_liberal = track_is_outside_timestep_range(
                    ts_start, ts_end, "liberal",
                    ts_start_track=ts_start_track,
                    ts_end_track=ts_end_track,
                )

        elif timesteps is not None:

            # Check if all timesteps of the track are in the list
            skip_track_strict = not tss_track.issubset(timesteps)

            # Check if any timesteps of the track are in the list
            skip_track_liberal = not (tss_track & timesteps)

        if tss_excl_mode_tracks == "strict" and skip_track_strict:
            # Skip tracks partially or fully outside timestep range,
            # along with all respective features
            tids_skip.add(tid)
            fids_skip.update(graph.vs["feature_id"])

        elif tss_excl_mode_tracks == "liberal":
            if skip_track_liberal:
                # Track is fully outside timesteps range; skip it
                tids_skip.add(tid)
                fids_skip.update(graph.vs["feature_id"])

            elif (skip_track_strict and
                    tss_excl_mode_tracked_features == "strict"):
                # Track is partially outside timesteps range; retain it
                # However, skip all features outside timesteps range
                if (ts_start, ts_end) != (None, None):
                    vss = [
                            graph.vs.select(ts_lt=ts_start),
                            graph.vs.select(ts_gt=ts_end),
                        ]
                elif timesteps:
                    vss = [graph.vs.select(ts_notin=timesteps)]
                for vs in vss:
                    fids_skip.update(vs["feature_id"])
    if not silent:
        if (ts_start, ts_end) != (None, None):
            _ts_start_str = str(ts_start) if ts_start else "???"
            _ts_end_str = str(ts_end) if ts_end else "???"
        elif timesteps:
            _ts_start_str = str(min(timesteps))
            _ts_end_str = str(max(timesteps))
        else:
            _ts_start_str, _ts_end_str = ["???"]*2
        period_str = "{}..{}".format(_ts_start_str, _ts_end_str)
        nsel_tracks = ntot_tracks - len(tids_skip)
        if nsel_tracks == ntot_tracks:
            sel_tracks_str = "{:6,}".format(ntot_tracks)
        else:
            sel_tracks_str = "{:6,}/{:6,}".format(nsel_tracks, ntot_tracks)
        nsel_features = ntot_features - len(fids_skip)
        if nsel_features == ntot_features:
            sel_features_str = "{:6,}".format(ntot_features)
        else:
            sel_features_str = "{:6,}/{:6,}".format(nsel_features, ntot_features)
        w, h = os.get_terminal_size()
        print(" "*w, end='\r', flush=True)
        print("[{}] read {} tracks and {} features".format(
                period_str, sel_tracks_str, sel_features_str))

    output = dict(
            graphs_by_tid  = graphs_by_tid,
            tids_skip      = tids_skip,
            fids_skip      = fids_skip,
        )
    if format == "extended":
        output.update(dict(
                header        = header,
                info_tracks   = info_tracks,
                info_features = info_features,
            ))
    return output

# SR_TMP <
def jdat_remove_noindent(jdat):
    for key, val in jdat.items():
        if isinstance(val, dict):
            jdat_remove_noindent(val)
        elif isinstance(val, list):
            for element in val:
                if isinstance(element, dict):
                    jdat_remove_noindent(element)
        elif isinstance(val, NoIndent):
            jdat[key] = val.value
# SR_TMP >

def __tmp__rebuild_tracks_from_graphs(
        infile,
        *,
        feature_name,

        read_pixelfile,
        read_pixels,
        rebuild_pixels,
        rebuild_pixels_if_necessary,
        graph_file_format,

        minsize,
        maxsize,
        timesteps,
        ts_start,
        ts_end,
        tss_excl_mode_tracks,
        tss_excl_mode_tracked_features,

        read_tracks,
        retain_n_biggest_tracks,
        optimize_pixelfile_input,
        timestep,

        silent,
        counter,

        ignore_missing_neighbors,
        ignore_edges_pshare_0,
        ignore_missing_total_track_stats,
        ignore_missing_missing_features_stats,
    ):

    # SR_TMP <
    if read_pixels and not rebuild_pixels_if_necessary:
        raise NotImplementedError("read_pixels")
    if retain_n_biggest_tracks and retain_n_biggest_tracks >= 0:
        raise NotImplementedError("retain_n_biggest_tracks")
    if optimize_pixelfile_input:
        raise NotImplementedError("optimize_pixelfile_input")
    # SR_TMP >

    # Read graphs containing track and feature info
    _r = read_track_graphs(infile, format=graph_file_format,
            timesteps=timesteps, ts_start=ts_start, ts_end=ts_end)
    graphs_by_tid = _r["graphs_by_tid"]
    header        = _r["header"]
    info_tracks   = _r["info_tracks"]
    info_features = _r["info_features"]
    tids_skip     = _r["tids_skip"]
    fids_skip     = _r["fids_skip"]

    # Extract some necessary meta data
    store_feature_values = header["store_feature_values"]
    separate_pixels_file = header["separate_pixels_file"]
    if separate_pixels_file:
        pixels_file = header["pixels_file"]
        # SR_TMP <
        pixels_file = "{}/{}".format(
                os.path.dirname(os.path.abspath(infile)),
                os.path.basename(pixels_file))
        # SR_TMP >

    # SR_TMP <
    read_feature_values = False
    # SR_TMP >

    if read_feature_values and not store_feature_values:
        print("warning: cannot read feature values from {}".format(infile))
        read_feature_values = False

    # SR_TMP <
    jdat_features = {
            "header"                : header,
            "info_"+feature_name    : info_features,
        }
    # SR_TMP >

    # Rebuild tracks and features
    if not separate_pixels_file:
        shells_by_fid = {}
        holes_by_fid = {}
        if read_feature_values:
            values_by_fid = {}
    tracks = []
    features = []
    for tid, graph in graphs_by_tid.items():
        if tid in tids_skip:
            continue
        jdat_track = graph["track_json_dict"]
        del graph["track_json_dict"]

        features_by_id = {}

        # Collect info to rebuild features
        timesteps_track = set()
        timesteps_by_fid = {}
        jdat_new_features = []
        for vx in graph.vs:
            fid = vx["feature_id"]

            jdat_new_features.append(vx["feature_json_dict"])

            timestep = vx["feature_json_dict"]["timestep"]
            timesteps_track.add(timestep)

            if not separate_pixels_file:
                # SR_TMP <
                try:
                    shells_by_fid[fid] = vx["feature_shells"]
                except KeyError:
                    # Old file (before multiple shells per feature)
                    shells_by_fid[fid] = vx["feature_shell"]
                # SR_TMP >
                holes_by_fid[fid] = vx["feature_holes"]
                if read_feature_values:
                    values_by_fid[fid] = vx["feature_values"]

        timesteps_track = sorted(timesteps_track)

        del graph.vs["feature_json_dict"]
        if not separate_pixels_file:
            # SR_TMP <
            try:
                del graph.vs["feature_shells"]
            except KeyError:
                # Old file (before multiple shells per feature)
                del graph.vs["feature_shell"]
            # SR_TMP <>
            del graph.vs["feature_holes"]
            if store_feature_values:
                del graph.vs["feature_values"]

        # Update features
        jdat_features["features_"+feature_name] = jdat_new_features

        # Rebuild features
        _r = rebuild_features(
                pixelfile                    = None,
                jdat                         = jdat_features,
                feature_name                 = feature_name,
                indir                        = None,
                timestep                     = None,
                read_pixels                  = False, # --------------------
                rebuild_pixels               = False, # pixels handled below
                rebuild_pixels_if_necessary  = False, # --------------------
                minsize                      = minsize,
                maxsize                      = maxsize,
                ts_start                     = ts_start,
                ts_end                       = ts_end,
                fids_skip                    = fids_skip,
                ignore_missing_neighbors     = ignore_missing_neighbors,
                counter                      = counter,
                silent                       = silent,
            )
        nskip_features  = _r["nskip_features"]
        new_features    = _r["features"]
        fids_skip       = _r["fids_skip"]

        features_by_id.update({f.id: f for f in new_features})
        features.extend(new_features)

        # Add pixels data to features
        if not separate_pixels_file:
            # Use pixels data extracted from graphs above
            for fid, feature in features_by_id.items():
                feature.set_shells(shells_by_fid.pop(fid))
                feature.set_holes(holes_by_fid.pop(fid))
                if read_feature_values:
                    feature.set_values(values_by_fid.pop(fid))
        elif read_pixelfile:
            # Read pixels data from separate file
            with h5py.File(pixels_file, "r") as fi:
                for fid, feature in features_by_id.items():
                    grp = fi[str(tid)][str(fid)]

                    shells = None
                    holes = []
                    values = None
                    for key in grp.keys():
                        # SR_TMP < Account for old files (one shell per feature)
                        if key == "shell":
                            shells = [grp[key][:]]
                        elif key == "shells":
                            shells = grp[key][:]
                        # SR_TMP >
                        elif key.startswith("hole_"):
                            holes.append(grp[key][:])
                        elif key == "values" and read_feature_values:
                            values = grp[key][:]
                    if shells is None:
                        err = "shells missing for feature {}/{}".format(tid, fid)
                        raise Exception(err)

                    feature.set_shells(shells)
                    feature.set_holes(holes)

                    if rebuild_pixels or rebuild_pixels_if_necessary:
                        nx, ny = header["nx"], header["ny"]
                        feature.derive_pixels_from_boundaries(nx, ny)

                    if read_feature_values:
                        if values is None:
                            err = "values missing for feature {}/{}".format(
                                    tid, fid)
                            raise Exception(err)
                        feature.set_values(values)

        # Rebuild track
        _r = rebuild_tracks(
                jdat_tracks          = [jdat_track],
                features_by_id       = features_by_id,
                graphs_tid           = {tid: graph},
                track_config         = info_tracks,
                ts_start             = ts_start,
                ts_end               = ts_end,
                tss_excl_mode_tracks = tss_excl_mode_tracks,
                tids_skip            = tids_skip,
                fids_skip            = fids_skip,
                debug                = False,
                counter              = counter,
                ignore_missing_total_track_stats      = ignore_missing_total_track_stats,
                ignore_missing_missing_features_stats = ignore_missing_missing_features_stats,
            )
        new_tracks   = _r["tracks"]
        nskip_tracks = _r["nskip_tracks"]
        tracks.extend(new_tracks)

    return dict(
            tracks         = tracks,
            features       = features,
            nskip_tracks   = nskip_tracks,
            nskip_features = nskip_features,
            info_tracks    = info_tracks,
            info_features  = info_features,
            header         = header,
        )


def rebuild_features(*,
        pixelfile,
        jdat,
        feature_name,
        indir,
        timestep,
        read_pixels,
        rebuild_pixels,
        rebuild_pixels_if_necessary,
        minsize,
        maxsize,
        ts_start,
        ts_end,
        fids_skip,
        ignore_missing_neighbors,
        counter,
        silent=False,
    ):

    if minsize is None:
        minsize = 1
    if maxsize is None:
        maxsize = -1
    if fids_skip is None:
        fids_skip = set()

    # Extract header information
    nx = jdat["header"].get("nx")
    ny = jdat["header"].get("ny")

    # Collect timesteps
    timesteps = set()
    target_key = "features_"+feature_name
    for key, val in jdat.items():
        if key == target_key:
            for jdat_feature in val:
                timesteps.add(jdat_feature["timestep"])
            break
    else:
        err = "missing '{}' feature data ({}) among {}".format(
                feature_name, target_key, sorted(jdat.keys()))
        raise Exception(err)
    timesteps = sorted(timesteps)
    period = "{}..{}".format(min(timesteps), max(timesteps))

    # Extract pixel store mode (and check consistency)
    pixel_store_mode = None
    names_all = []
    for key, val in jdat.items():
        if key.startswith("info_") and not key == "info_tracks":
            name = key.replace("info_", "")
            names_all.append(name)
            try:
                 mode = val["pixel_store_mode"]
            except KeyError:
                err = "pixel store mode not found for {}".format(name)
                raise Exception(err)
            if pixel_store_mode is None:
                pixel_store_mode = mode
            elif mode != pixel_store_mode:
                err = ("features have different pixel store modes: {} != {}"
                        ).format(mode, pixel_store_mode)
                raise Exception(err)
    # SR_TMP <
    assert len(names_all) == 1, "more than one feature type"
    # SR_TMP >
    if pixel_store_mode == "boundaries":
        if rebuild_pixels_if_necessary:
            rebuild_pixels = True
        if read_pixels:
            if not rebuild_pixels_if_necessary:
                log.warning("features stored in 'boundaries' mode; "
                        "override read_pixels=True with read_pixels=False")
            read_pixels = False

    # Determine which features to skip based on size, timestep, ...
    key_starts_skip = []
    if (minsize > 1 or maxsize > 0 or
            ts_start is not None or ts_end is not None):
        for key, jdat_i in jdat.items():
            if not key.startswith("features_"):
                continue
            fname = key.split("_", 1)[1]
            for jdat_feature in jdat_i:
                fid = jdat_feature["id"]
                key_feature = "{}_{}".format(fname, jdat_feature["id"])

                # Check if skipping pre-determined
                if fid in fids_skip:
                    key_starts_skip.append(key_feature)
                    continue

                # Check minimum size
                if (minsize > 1 and
                        jdat_feature["stats"]["n"] < minsize):
                    key_starts_skip.append(key_feature)
                    fids_skip.add(fid)
                    continue

                # Check maximum size
                if (maxsize > 0 and
                        jdat_feature["stats"]["n"] > maxsize):
                    key_starts_skip.append(key_feature)
                    fids_skip.add(fid)
                    continue

                # Check timestep
                _ts = jdat_feature["timestep"]
                if (ts_start is not None and _ts < ts_start or
                        ts_end is not None and _ts > ts_end):
                    key_starts_skip.append(key_feature)
                    fids_skip.add(fid)
                    continue

    nskip_features = len(key_starts_skip)

    # Read pixels
    pixel_tables = None
    if pixelfile is not None:
        pixel_tables = read_feature_pixels(
                pixelfile, [feature_name],
                nx=nx, ny=ny,
                mode            = pixel_store_mode,
                read_pixels     = read_pixels,
                rebuild_pixels  = rebuild_pixels,
                minsize         = minsize,
                maxsize         = maxsize,
                counter         = counter,
                fids_skip       = fids_skip,
                key_starts_skip = key_starts_skip,
                silent          = silent,
                period          = period,
            )

    # Rebuild features
    features = rebuild_features_core(
            jdat, feature_name,
            pixel_store_mode            = pixel_store_mode,
            pixel_tables                = pixel_tables,
            read_pixels                 = read_pixels,
            fids_skip                   = fids_skip,
            pixelfile                   = pixelfile,
            ignore_missing_neighbors    = ignore_missing_neighbors,
            pixels_missing              = (not read_pixels),
            counter                     = counter,
            period                      = period,
        )

    # SR_TODO move rebuild block somewhere where it's called for optimized pixelfile input
    if not read_pixels and rebuild_pixels:
        for feature in features:
            try:
                feature.derive_pixels_from_boundaries(nx, ny)
            except Exception:
                err = "feature {}: rebuild of pixels failed".format(
                        feature.id)
                raise Exception(err)

    # Set timestep if given
    if timestep is not None:
        for feature in features:
            feature.timestep = timestep

    return dict(
            features        = features,
            nskip_features  = nskip_features,
            timesteps       = timesteps,
            fids_skip       = fids_skip,
        )

def features_read_pixels(feature_name, features, pixelfiles_fids, jdat, *,
        read_pixels, rebuild_pixels, rebuild_pixels_if_necessary,
        counter, fids_select=None, fids_skip=None, period=None,
    ):
    """Read necessary pixelfiles and add pixels etc. to features."""

    nx = jdat["header"]["nx"]
    ny = jdat["header"]["ny"]

    log.info(("restore pixels for {:,} features from {:,} pixelfiles"
            ).format(len(features), len(pixelfiles_fids)))

    # SR_TMP <
    pixel_store_mode = jdat["info_"+feature_name]["pixel_store_mode"]
    # SR_TMP >

    ntot = len(features)
    itot = 0
    for pixelfile, fids_file in sorted(pixelfiles_fids.items()):

        if counter:
            msg = " {: 2.0%} {}".format((0 if ntot == 0 else itot/ntot),
                    pixelfile)
            try:
                w, h = os.get_terminal_size()
            except OSError:
                pass
            else:
                print(msg[:w], end="\r", flush=True)

        # Select features
        if fids_select is None:
            fids_i = fids_file
        else:
            fids_i = [fid for fid in fids_file if fid in fids_select]
        features_i = [f for f in features if f.id in fids_i]
        fids_i = [f.id for f in features_i]
        if len(fids_i) == 0:
            # Nothing to do!
            continue

        itot += len(fids_i)

        pixel_tables = read_feature_pixels(
                pixelfile, [feature_name],
                nx=nx, ny=ny,
                mode            = pixel_store_mode,
                read_pixels     = read_pixels,
                rebuild_pixels  = rebuild_pixels,
                # minsize         = minsize,
                # maxsize         = maxsize,
                counter         = True,
                fids_select     = fids_i,
                fids_skip       = fids_skip,
                # key_starts_skip = key_starts_skip,
                silent          = True,
                period          = period,
            )

        # Organize tables data
        arrs_fid = {}
        for key, arr in pixel_tables.items():
            name, sfid, type_ = key.split("_", 2)
            fid = int(sfid)
            if fid not in arrs_fid:
                arrs_fid[fid] = {}
            if type_.startswith("hole"):
                if "holes" not in arrs_fid[fid]:
                    arrs_fid[fid]["holes"] = []
                arrs_fid[fid]["holes"].append(arr)
            else:
                arrs_fid[fid][type_] = arr

        # Check feature ids
        if set(fids_i) != arrs_fid.keys():
            err = ("reading pixelfile {} failed: differing feature ids"
                    ).format(pixelfile)
            _missing = set(fids_i).difference(arrs_fid.keys())
            if _missing:
                err += "\n -> {} missing: {}".format(len(_missing),
                        ", ".join([str(i) for i in _missing]))
            _toomany = set(arrs_fid.keys()).difference(fids_i)
            if _toomany:
                err += "\n -> {} too many: {}".format(len(_toomany),
                        ", ".join([str(i) for i in _toomany]))
            raise Exception(err)

        # Add arrays to features
        for feature in features_i:
            fid = feature.id

            pixels = arrs_fid[fid].get("pixels")
            feature.set_pixels(pixels)

            # SR_TMP <
            try:
                shells  = arrs_fid[fid].get("shells")
            except KeyError:
                # Old file (before multiple shells per feature)
                shells  = [arrs_fid[fid].get("shell")]
            # SR_TMP >
            feature.set_shells(shells)

            values = arrs_fid[fid].get("values")
            if values is not None:
                feature.set_values(values)

            holes = arrs_fid[fid].get("holes")
            if holes is not None:
                feature.set_holes(holes)

    if counter:
        try:
            w, h = os.get_terminal_size()
        except OSError:
            pass
        else:
            print(" "*w, end="\r", flush=True)

def rebuild_tracks(*, jdat_tracks, features_by_id, graphs_tid, track_config,
        ts_start, ts_end, tss_excl_mode_tracks, debug, counter, tids_skip,
        fids_skip,
        ignore_missing_total_track_stats, ignore_missing_missing_features_stats,
    ):

    if debug: log.debug("rebuild {} tracks".format(len(jdat_tracks)))

    if tids_skip is None:
        tids_skip = []
    if fids_skip is None:
        fids_skip = []

    if counter:
        period = "{}..{}".format(
                min(jdat_tracks[ 0]["timesteps"]),
                max(jdat_tracks[-1]["timesteps"]))

    ntot = len(jdat_tracks)
    nskip_tracks = 0
    ni, di = 0, np.ceil(float(ntot)/100)
    tracks = []
    for jdat_track in jdat_tracks:
        tid = jdat_track["id"]
        ni += 1
        if debug: log.debug(" ({}/{}) {}".format(ni, ntot, tid))
        if counter and ni%di == 0:
            msg = ".. {: 2.0%} rebuilding tracks ({})".format(ni/ntot, period)
            try:
                w, h = os.get_terminal_size()
            except OSError:
                pass
            else:
                print(msg[:w], end="\r", flush=True)

        # Skip track if mandated
        if tids_skip is not None and tid in tids_skip:
            nskip_tracks += 1
            continue

        # Skip track if outside timestep range
        if track_is_outside_timestep_range(ts_start, ts_end,
                tss_excl_mode_tracks, jdat=jdat_track):
            nskip_tracks += 1
            continue

        graph = graphs_tid[tid]

        # Link features to respective vertices
        features_track = []
        for vertex in graph.vs[:]:
            fid = vertex["feature_id"]
            if fid in fids_skip:
                # Skip feature
                continue
            feature_vertex = features_by_id[fid]
            vertex["feature"] = feature_vertex
            feature_vertex.set_vertex(vertex)
            features_track.append(feature_vertex)
        if all(fid is None for fid in graph.vs["feature_id"]):
            del graph.vs["feature_id"]

        # Create track
        track = FeatureTrack(
                id_      = tid,
                graph    = graph,
                features = features_track,
                config   = track_config,
            )

        tracks.append(track)

        if not track.is_complete():
            # -- Handle partial tracks: store missing stats

            # Total track stats
            stats = jdat_track["total_track_stats"]
            if not stats and not ignore_missing_total_track_stats:
                err = ("track {}: incomplete, but no total track stats"
                        ).format(track.id)
                raise Exception(err)
            track.set_total_track_stats(stats,
                    ignore_missing=ignore_missing_total_track_stats)

            # Missing features stats
            stats = jdat_track["missing_features_stats"]
            if not stats and not ignore_missing_missing_features_stats:
                err = ("track {}: incomplete, but no missing features stats"
                        ).format(track.id)
                raise Exception(err)
            track.set_missing_features_stats(stats,
                    ignore_missing=ignore_missing_missing_features_stats)

    if counter:
        try:
            w, h = os.get_terminal_size()
        except OSError:
            pass
        else:
            print(" "*w, end="\r", flush=True)

    return dict(
            tracks       = tracks,
            nskip_tracks = nskip_tracks,
        )

def track_is_outside_timestep_range(ts_start, ts_end, tss_excl_mode_tracks, *,
        jdat=None, ts_start_track=None, ts_end_track=None):
    """Check whether to skip a track based on its timestep range."""

    if jdat is not None:
        _tss = jdat["timesteps"]
        if "missing_features_stats" in jdat:
            _tss.extend(jdat["missing_features_stats"]["timestep"])
        ts_start_track = min(_tss)
        ts_end_track = max(_tss)

    skip_track = False
    if ts_start is not None or ts_end is not None:
        # Check whether timestep is out of range
        if tss_excl_mode_tracks == "strict":
            if (ts_start is not None and ts_start_track < ts_start or
                    ts_end is not None and ts_end_track > ts_end):
                skip_track = True

        elif tss_excl_mode_tracks == "liberal":
            if (ts_start is not None and ts_end_track < ts_start or
                    ts_end is not None and ts_start_track > ts_end):
                skip_track = True

        else:
            raise ValueError("invalid tss_excl_mode_tracks: "+tss_excl_mode_tracks)

    return skip_track

def read_feature_pixels(pixelfile, *args, **kwas):
    """Read feature pixels from npz archive (including shells and holes).

    In 'pixels' mode, all pixels are read from the archive, whereas in
    'boundaries' mode, only the shells and holes pixels are read and the
    interios pixels reconstructed from the boundaries. In 'boundaries' mode
    pixels have no values.
    """
    debug = False
    if debug: log.debug("< read_feature_pixels from {}".format(pixelfile))

    if pixelfile.endswith(".npz"):
        with np.load(pixelfile) as fi:
            return _read_feature_pixels_core(fi, *args, **kwas)
    elif pixelfile.endswith(".h5"):
        with h5py.File(pixelfile, "r") as fi:
            return _read_feature_pixels_core(fi, *args, **kwas)

def _read_feature_pixels_core(fi, names, *, mode, nx=None, ny=None,
        read_pixels=True, rebuild_pixels=True, minsize=1, maxsize=-1,
        counter=False, period=None, fids_select=None, fids_skip=None,
        key_starts_skip=None, silent=False,
    ):

    # SR_TMP <
    if counter and period is None:
        raise ValueError("must pass period for counter")
    # SR_TMP >

    if maxsize == 0:
        raise ValueError("invalid maxsize 0")

    # Check validity mode
    if mode not in ("pixels", "boundaries"):
        err = "invalid mode {}".format(mode)
        raise ValueError(err)
    if mode == "boundaries" and  (nx is None or ny is None):
        err = "'boundaries' mode requires nx and ny"
        raise ValueError(err)

    # Sort keys by variable name
    keys_name = {}
    for name in names: # SR_TODO eliminate multiple names per file
        keys_name[name] = [key for key in fi.keys() if key.startswith(name)]

    # Process variables one-by-one
    if mode == "pixels":
        ntot = len(fi.keys())
    elif mode == "boundaries":
        # SR_TMP <
        ntot = len([k for k in fi.keys() if k.endswith("_shells_0")])
        if ntot == 0:
            # Most likely old file (only one shell per feature)
            ntot = len([k for k in fi.keys() if k.endswith("_shell")])
        # SR_TMP >
    ni, di = 0, np.ceil(float(ntot)/100)
    pixel_tables = {}

    if fids_select:
        fids_select_str = ["_{}_".format(fid) for fid in fids_select]
    if fids_skip:
        fids_skip_str = ["_{}_".format(fid) for fid in fids_skip]

    if key_starts_skip:
        key_starts_n = len(next(iter(key_starts_skip)))

    for name in names: # SR_TODO eliminate multiple names per file
        for key in fi.keys():
            ni += 1
            if counter and ni%di == 0:
                msg = ".. {: 2.0%} reading pixels ({})".format(ni/ntot, period)
                try:
                    w, h = os.get_terminal_size()
                except OSError:
                    pass
                else:
                    print(msg[:w], end="\r", flush=True)

            if key_starts_skip and key[:key_starts_n] in key_starts_skip:
                continue

            if (fids_select and not any(s in key for s in fids_select_str)):
                continue

            if (fids_skip and any(s in key for s in fids_skip_str)):
                continue

            if read_pixels:
                if (    key.endswith("_pixels") or
                        key.endswith("_values") or
                        "_shell_" in key        or
                        # SR_TMP < Account for old files (one shell per feature)
                        key.endswith("_shell")  or
                        # SR_TMP >
                        "_hole_" in key         ):
                    pixel_tables[key] = fi[key][:]
            else:
                if (    "_shell_" in key        or
                        # SR_TMP <
                        key.endswith("_shell")  or
                        # SR_TMP >
                        "_hole_" in key         ):
                    pixel_tables[key] = fi[key][:]

    if counter:
        try:
            w, h = os.get_terminal_size()
        except OSError:
            pass
        else:
            print(" "*w, end="\r", flush=True)

    return pixel_tables

def rebuild_features_core(
        jdat,
        feature_name,
        *,
        pixel_store_mode,
        pixel_tables,
        read_pixels,
        fids_skip,
        pixelfile,
        ignore_missing_neighbors,
        pixels_missing,
        counter,
        period,
    ):
    debug = False
    if debug: log.debug("< rebuild_features_core")

    features = []
    features_by_id = {}
    neighbors_ids = {}

    def name2key(name):
        return "features" if name == "" else "features_{}".format(name)

    ntot = len(jdat.get(name2key(feature_name), []))
    n_rebuild = ntot - len(fids_skip)
    ni, di = 0, np.ceil(float(ntot)/100)
    if debug: log.debug("rebuild features {}".format(feature_name))
    try:
        jdat_features = jdat.pop(name2key(feature_name))
    except KeyError as e:
        err = "features not found in file: {}".format(name2key(feature_name))
        raise Exception(err) from e

    for jdat_feature in jdat_features:
        fid = jdat_feature["id"]
        ni += 1
        if debug: log.debug(" ({}/{}) {}".format(ni, ntot, jdat_feature["id"]))
        if counter and ni%di == 0:
            msg = ".. {:3.0%} rebuilding {:,}/{:,} features ({})".format(
                    ni/ntot, n_rebuild, ntot, period)
            try:
                w, h = os.get_terminal_size()
            except OSError:
                pass
            else:
                print(msg[:w], end="\r", flush=True)

        # Check whether to skip the feature
        if fid in fids_skip:
            continue

        # Create feature
        feature = Feature.from_jdat(
                jdat_feature,
                name            = feature_name,
                pixel_tables    = pixel_tables,
                pixels_missing  = pixels_missing,
            )

        # Store neighbor IDs
        features_by_id[feature.id] = feature
        neighbors_ids[feature.id] = jdat_feature.get("neighbors", [])

        # If pixels are not read, store the name of the pixelfile
        if not read_pixels:
            feature.properties["pixelfile"] = pixelfile

        features.append(feature)

    if counter:
        try:
            w, h = os.get_terminal_size()
        except OSError:
            pass
        else:
            print(" "*w, end="\r", flush=True)

    # Link neighbors
    if debug: log.debug("link neighbors")
    for feature in features_by_id.values():
        if debug: log.debug(" -> feature {}".format(feature.id))
        for fid in neighbors_ids[feature.id]:
            try:
                neighbor = features_by_id[fid]
            except KeyError:
                if not ignore_missing_neighbors:
                    log.warning(("cannot assign feature {} as neighbor "
                        "to feature {}: feature missing").format(
                        fid, feature.id))
            else:
                feature.neighbors.append(neighbor)

    # if counter:
    #    print("rebuild features : 100%", flush=True)

    if debug: log.debug("> rebuild_features_core")
    return features

def select_tracks_features(*,
        tracks                      = None,
        features                    = None,
        nskip_tracks                = None,
        nskip_features              = None,
        ts_start                    = None,
        ts_end                      = None,
        tss_excl_mode_tracks        = None,
        retain_n_biggest_tracks     = None,
        discard_untracked_features  = None,
        silent                      = False,
    ):

    nto = None if tracks is None else len(tracks)
    nfo = None if features is None else len(features)

    # Check arguments
    if tracks and nskip_tracks is None:
        raise ValueError("must pass nskip_tracks alongside tracks")
    if features and nskip_features is None:
        raise ValueError("must pass nskip_features alongside features")
    if ts_start is not None or ts_end is not None:
        if tss_excl_mode_tracks is None:
            raise ValueError("must pass tss_excl_mode_tracks alongside "
                    "ts_start and/or ts_end")

    if tracks and retain_n_biggest_tracks:
        # Pick all subtracks of the N biggest tracks
        sizes_tid = {}
        tracks_tid = {}
        for track in tracks:
            # SR_TMP <
            if track.id in sizes_tid:
                assert sizes_tid[track.id] == track.size(total=True)
            # SR_TMP >
            sizes_tid[track.id] = track.size(total=True)
            if track.id not in tracks_tid:
                tracks_tid[track.id] = []
            tracks_tid[track.id].append(track)
        biggest_tids = sorted([(n, tid) for tid, n in sizes_tid.items()],
                reverse=True)[:retain_n_biggest_tracks]
        tracks_sel = [t for n, tid in biggest_tids for t in tracks_tid[tid]]
        for track in tracks:
            if track not in tracks_sel:
                nskip_tracks += 1
                track.unlink_features()
        tracks = tracks_sel

    # SR_TMP < TODO figure out if this makes sense...
    if tracks or nskip_tracks == 0:
    # SR_TMP >
        if tss_excl_mode_tracks == "liberal" and (
                ts_start is not None or ts_end is not None):
            # Remove all features outside the target period
            # that don't belong to a track
            for feature in [f for f in features]:
                if feature.track() is None:
                    _ts = feature.timestep
                    if (ts_start is not None and _ts < ts_start or
                            ts_end is not None and _ts > ts_end):
                        features.remove(feature)
                        nskip_features += 1

    if discard_untracked_features:
        # Only keep features linked to a track
        for feature in [f for f in features]:
            if feature.track() is None:
                nskip_features += 1
                features.remove(feature)

    if not silent:
        msg = "selected"
        if features is not None:
            nfn = len(features)
            nftn = len([f for f in features if f.track() is not None])
            msg += " {:,}/{:,} features ({:,} tracked)".format(nfn, nfo, nftn)
            if tracks is not None:
                msg += " and"
        if tracks is not None:
            ntn = None if tracks is None else len(tracks)
            msg += " {:,}/{:,} tracks".format(ntn, nto)
        log.info(msg)

    return dict(
            tracks         = tracks,
            features       = features,
            nskip_tracks   = nskip_tracks,
            nskip_features = nskip_features,
        )


def read_masks(infile, lon, lat, silent=False, dtype=bool):
    """Read mask shells and holes and turn them into mask fields."""

    with np.load(infile) as fi:
        paths = dict(fi.items())

    names = paths.keys()
    names = sorted(set([n[::-1].split("_", 1)[1][::-1] for n in names]))

    masks = {}
    if not silent:
        log.info("note: first masks takes a while (create cKDTree once)")
    for name in names:
        if not silent:
            log.info("create mask {}".format(name))

        # Collect lon/lat paths
        shells_ll = [v for k, v in paths.items() if k.startswith(name+"_s")]
        holes_ll  = [v for k, v in paths.items() if k.startswith(name+"_h")]

        # Convert lon/lat to indices
        shells_xy = []
        for shell_ll in shells_ll:
            pts_lon, pts_lat = shell_ll.T
            shell_xy = points_lonlat_to_inds(pts_lon, pts_lat, lon, lat)
            shells_xy.append(shell_xy)
        holes_xy = []
        for hole_ll in holes_ll:
            pts_lon, pts_lat = hole_ll.T
            hole_xy = points_lonlat_to_inds(pts_lon, pts_lat, lon, lat)
            holes_xy.append(hole_xy)

        # Create mask
        raster = PIL.Image.new("L", lon.shape, 0)
        for shell in shells_xy:
            shell = [(x, y) for x, y in shell]
            PIL.ImageDraw.Draw(raster).polygon(shell, fill=1, outline=1)
        for hole in holes_xy:
            hole = [(x, y) for x, y in hole]
            PIL.ImageDraw.Draw(raster).polygon(hole, fill=0, outline=0)
        mask = np.array(raster, dtype).T
        masks[name] = mask

    return masks
