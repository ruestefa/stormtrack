# -*- coding: utf-8 -*-
"""
Front surgery.

Algorithms to split, categorize, and remerge front fragments, developed in
order to separate large-scale fronts from small-scale noise.

Ultimately not a success, but retained given it's good test coverage that
complement and add to the tests of ``stormtrack.core``.

"""
from __future__ import print_function

# C: C libraries
from libc.stdlib cimport exit
from libc.stdlib cimport free
from libc.stdlib cimport malloc

# C: Third-party
cimport numpy as np

# Standard library
import logging as log
import os
import re
import sys
import time
from pprint import pformat

# Third-party
import numpy as np

# Local
from ..core.identification import identify_features
from ..core.io import read_feature_file
from ..core.io import write_feature_file
from ..utils.netcdf import nc_read_var_list


cdef list CATEGORIES = ["clutter0", "clutter1", "small", "medium", "large"]
cdef int N_CATEGORIES = len(CATEGORIES)


cpdef void run_front_surgery_main(
    list timesteps,
    dict conf_general,
    dict conf_grid,
    dict conf_identification,
    dict conf_surgery,
) except *:
    cdef bint debug = False
    cdef int i
    cdef int j
    cdef int k
    cdef int n

    # Add dummy timesteps to end of timesteps list to add two iterations to
    # the subsequent loop, because surgery is only complete after two
    # iterations and processed features are only written to disk two
    # iterations after they have been read
    cdef int n_timesteps = len(timesteps)
    timesteps.extend([None]*2)

    # Set up constants
    cdef cConstants constants = cConstants(
        nx=conf_grid["nx"],
        ny=conf_grid["ny"],
        connectivity=conf_grid["connectivity"],
        n_neighbors_max=conf_grid["n_neighbors_max"],
    )

    # Categories defined globally above
    # (This allows to access them for print statements
    # without having to pass them around all over the place)
    global N_CATEGORIES
    cdef int n_categories = N_CATEGORIES
    assert len(conf_surgery["thresholds"]) == 2

    # Prepare cregions
    cdef cRegions* cfronts_now=NULL
    cdef cRegions* cfronts_new=NULL
    cdef cRegions* cfronts_old=NULL
    cdef cRegions* cfronts_tmp = NULL

    # Prepare category size arrays
    cdef int* categories_new_n = NULL
    cdef int* categories_now_n = NULL
    cdef int* categories_old_n = NULL
    cdef int* categories_tmp_n = NULL

    cdef dict fronts_old
    cdef dict fronts_now
    cdef dict fronts_new
    cdef list fronts_raw

    # Initialize grids
    cdef np.ndarray[np.float32_t, ndim=2] field_raw = np.zeros(
        [constants.nx, constants.ny], dtype=np.float32,
    )
    cdef cGrid grid0 = grid_create_empty(constants)
    cdef cGrid grid1 = grid_create_empty(constants)
    cdef cGrid grid2 = grid_create_empty(constants)
    cdef cGrid* grid_new = &grid0
    cdef cGrid* grid_now = NULL
    cdef cGrid* grid_old = NULL

    # Allocate lookup tables
    pixel_status_table_alloc(&grid_new.pixel_status_table, &constants)
    pixel_status_table_alloc(&grid_now.pixel_status_table, &constants)
    pixel_status_table_alloc(&grid_old.pixel_status_table, &constants)
    neighbor_link_stat_table_alloc(&grid_new.neighbor_link_stat_table, &constants)
    neighbor_link_stat_table_alloc(&grid_now.neighbor_link_stat_table, &constants)
    neighbor_link_stat_table_alloc(&grid_old.neighbor_link_stat_table, &constants)

    # Extract some often-used config parameters
    cdef str feature_name_base = conf_general["feature_name"]
    cdef str input_format = conf_general["input_format"]
    cdef str infile_format = conf_general["infile_format"]
    cdef str varname = conf_identification["varname"]
    cdef int trim_bnd_n = conf_identification["trim_boundaries_n"]
    cdef int output_skip_n = conf_general.get("output_skip_n", 0)

    # Construct info block in case of netcdf input
    cdef dict info_dict = None
    cdef dict info_dict_prev = None
    if input_format.startswith("netcdf"):
        # SR_TODO equivalent code block also in identify_features.py
        # SR_TODO remove duplicate code!
        # Construct info dict
        info_dict = dict(varname=feature_name_base)
        info_dict["thresholds"] = [conf_identification["threshold"], None]
        _keys = [
            "minsize",
            "split_levels",
            "split_seed_minsize",
            "split_seed_minstrength",
            "topo_filter_apply",
            "topo_filter_mode",
            "topo_filter_min_overlap",
        ]
        for key in _keys:
            info_dict[key] = conf_identification[key]
        _keys = ["thresholds", "min_overlap", "max_size_temporal_step"]
        # _keys += ["threshold_boundary_neighbors", "threshold_boundary_all"]
        for key in _keys:
            info_dict[f"surgery_{key}"] = conf_surgery[key]

    cdef str sts
    cdef Feature front
    cdef np.ndarray[np.float32_t, ndim=2] rlon = conf_grid.get("rlon")
    cdef np.ndarray[np.float32_t, ndim=2] rlat = conf_grid.get("rlat")
    cdef np.ndarray[np.float32_t, ndim=2] fld
    cdef dict timesteps_ts = dict(old=None, now=None, new=None)
    for timestep in timesteps:
        timesteps_ts["new"] = timestep

        fronts_raw = None
        if timestep is not None:

            # Reconstruct file name from timestep
            sts = str(timestep)
            infile = infile_format.format(
                YYYY=sts[:4], MM=sts[4:6], DD=sts[6:8], HH=sts[8:10],
            )

            # Identify features from raw field (full field)
            if input_format == "netcdf":
                raise NotImplementedError("input_format 'netcdf'")

            # Identify features from raw field (pixel list)
            elif input_format == "netcdf-list":

                # Read pixels and convert to 2d field
                if rlon is None or rlat is None:
                    err = "input_format 'netcdf-list': require rlon, rlat"
                    raise Exception(err)
                log.info(f"read field {varname} from file {infile}")
                fld = nc_read_var_list(infile, varname, rlon, rlat)
                if trim_bnd_n > 0:
                    # SR_TODO add value to which to trim to interface
                    fld[:trim_bnd_n, :] = 0
                    fld[:, :trim_bnd_n] = 0
                    fld[-trim_bnd_n:, :] = 0
                    fld[:, -trim_bnd_n:] = 0

                # Run identification
                base_id = timestep*1e5
                upper = -1
                fronts_raw = identify_features(
                    fld,
                    feature_name_base,
                    conf_grid["nx"],
                    conf_grid["ny"],
                    conf_identification["threshold"],
                    upper,
                    conf_identification["minsize"],
                    base_id,
                    timestep,
                    conf_identification["split_levels"],
                    conf_identification["split_seed_minsize"],
                    conf_identification["split_seed_minstrength"],
                    conf_identification["topo_filter_apply"],
                    conf_identification["topo_filter_mode"],
                    conf_identification["topo_mask"],
                    conf_identification["topo_filter_min_overlap"],
                )
                log.info("identified {len(fronts_raw)} {feature_name_base} features")

            # Read pre-identified features from file
            elif input_format == "json":

                # Read features from infile
                log.info(
                    f"\n[{timestep}] read {feature_name_base} features from {infile}"
                )
                fronts_raw, info_dict = import_fronts(
                    infile, feature_name_base, timestep, conf_grid,
                )

                # Check and extend info dict
                if len(info_dict) == 0:
                    log.warning("empty info_dict dict")
                if info_dict_prev is not None:
                    if info_dict != info_dict_prev:
                        raise Exception(
                            f"info dicts differ:\n\n"
                            f"NEW:\n{pformat(sorted(info_dict.items()))}\n\n"
                            f"NOW:\n{pformat(sorted(info_dict_prev.items()))}\n"
                        )
                info_dict_prev = info_dict.copy()
                _keys = ["thresholds", "min_overlap", "max_size_temporal_step"]
                # _keys += ["threshold_boundary_neighbors", "threshold_boundary_all"]
                for key in _keys:
                    info_dict[f"surgery_{key}"] = conf_surgery[key]

            else:
                raise Exception("invalid input_format", input_format)

            # Allocate new cregions if necessary
            if cfronts_new is NULL:
                if debug: log.debug("allocate new cregions")
                cfronts_new = <cRegions*>malloc(n_categories*sizeof(cRegions))
                for i in range(n_categories):
                    cfronts_new[i] = cregions_create(50)

            # Allocate new pixels if necessary
            if grid_new.pixels is NULL:
                if debug:
                    log.debug("allocate new pixels")
                grid_create_pixels(grid_new, field_raw)

            # Allocate new categories if necessary
            if categories_new_n is NULL:
                if debug:
                    log.debug("allocate new category size array")
                categories_new_n = <int*>malloc(n_categories*sizeof(int))
                for i in range(n_categories):
                    categories_new_n[i] = 0

        log.info(f"\n{'<' * 50}\n")
        _run_front_surgery_core(
            timesteps_ts,
            fronts_raw,
            cfronts_new,
            cfronts_now,
            cfronts_old,
            n_categories,
            categories_new_n,
            categories_now_n,
            categories_old_n,
            grid_new,
            grid_now,
            grid_old,
            &constants,
            conf_surgery,
        )
        log.info(f"\n{'<' * 50}\n")

        # Finish fully processed features and write them to disk
        silent = True
        jformat = conf_general["json_format"]
        if cfronts_old is not NULL:

            # Merge all non-clutter features
            merge_categories(
                cfronts_old,
                category_medium,
                category_large,
                n_categories,
                categories_old_n,
                &constants,
                neighbors_across_categories=False,
            )
            merge_features_in_categories(
                cfronts_old,
                n_categories,
                categories_old_n,
                grid_old,
                &constants,
                reset_connections=False,
            )

            # SR_TODO remove hard-coded limit (not really pressing, though)
            # Make sure we have enough id; in case of too many features, abort
            for i in range(n_categories):
                limit = int(1e5)
                if categories_old_n[i] >= limit:
                    log.error(
                        f"not enough ids for {categories_old_n[i]} features (only "
                        f"{int(1e5)})! increase hard-coded limit of {limit}!"
                    )
                    exit(4)
            base_id = timesteps_ts["old"]*1e5

            # Convert back to Feature objects
            if debug: log.debug(">>> cfronts_old -> fronts_old")
            fronts_old = cfronts2fronts(
                cfronts_old,
                n_categories,
                categories_old_n,
                base_id,
                timesteps_ts["old"],
                grid_old,
                &constants,
            )
            cfronts_old = NULL

            # Write output unless step in range output_skip_n from either end
            i = timesteps.index(timesteps_ts["old"])
            if output_skip_n <= i < (n_timesteps - output_skip_n):
                assert len(fronts_old) > 1, "not implemented: multiple front categories"
                (cat, fronts) = next(iter(fronts_old.items()))
                sts = str(timesteps_ts["old"])
                outfile = conf_general["outfile_format"].format(
                    YYYY=sts[:4], MM=sts[4:6], DD=sts[6:8], HH=sts[8:10],
                )
                feature_name = f"{feature_name_base}_{cat}"
                log.info(f"write categorized features to {outfile}")
                write_feature_file(
                    outfile,
                    feature_name     = feature_name,
                    features         = fronts,
                    info_features    =  info_dict,
                    nx               = conf_grid["nx"],
                    ny               = conf_grid["ny"],
                    jformat          = jformat,
                    jdat_old         = {},
                    pixel_store_mode = "pixels",
                    silent           = silent,
                )

            # Skip output because step in range output_skip_n from either end
            else:
                log.info(f"skip output for timestep {timesteps_ts['old']}")

        # Update "processing queue"
        _update_queue(
            &cfronts_new,
            &cfronts_now,
            &cfronts_old,
            &cfronts_tmp,
            n_categories,
            &categories_new_n,
            &categories_now_n,
            &categories_old_n,
            &categories_tmp_n,
            grid_new,
            grid_now,
            grid_old,
            timesteps_ts,
            &constants,
        )

    # SR_TODO Activate!!! Properly clean up all regions!!!
    # Clean up cregions
    if cfronts_new is not NULL:
        for i in range(n_categories):
            cregions_cleanup(&cfronts_new[i], cleanup_regions=True)
        free(cfronts_new)
        cfronts_new = NULL
    if cfronts_now is not NULL:
        for i in range(n_categories):
            cregions_cleanup(&cfronts_now[i], cleanup_regions=True)
        free(cfronts_now)
        cfronts_now = NULL
    if cfronts_old is not NULL:
        for i in range(n_categories):
            cregions_cleanup(&cfronts_old[i], cleanup_regions=True)
        free(cfronts_old)
        cfronts_old = NULL

    # Clean up grids
    grid_new = NULL
    grid_now = NULL
    grid_old = NULL
    grid_cleanup(&grid0)
    grid_cleanup(&grid1)
    grid_cleanup(&grid2)


cdef void _update_queue(
    cRegions** cfronts_new,
    cRegions** cfronts_now,
    cRegions** cfronts_old,
    cRegions** cfronts_tmp,
    int n_categories,
    int** categories_new_n,
    int** categories_now_n,
    int** categories_old_n,
    int** categories_tmp_n,
    cGrid* grid_new,
    cGrid* grid_now,
    cGrid* grid_old,
    dict timesteps_ts,
    cConstants* constants,
):
    cdef cGrid* grid_tmp

    # SR_TODO exchange the whold cGrid objects (not only the pixels)

    # NOW -> OLD
    if cfronts_now[0] is not NULL:
        if cfronts_old[0] is not NULL:
            cfronts_tmp[0] = cfronts_old[0]
            for i in range(n_categories):
                cfronts_tmp[0][i].n = 0
        cfronts_old[0] = cfronts_now[0]
        cfronts_now[0] = NULL
        if categories_old_n[0] is not NULL:
            categories_tmp_n[0] = categories_old_n[0]
            for i in range(n_categories):
                categories_tmp_n[0][i] = 0
        categories_old_n[0] = categories_now_n[0]
        categories_now_n[0] = NULL
        if grid_old is not NULL:
            grid_tmp = grid_old
            grid_reset(grid_tmp)
        grid_old = grid_now
        grid_now = NULL

    # NEW -> NOW
    if cfronts_new[0] is not NULL:
        cfronts_now[0] = cfronts_new[0]
        if cfronts_tmp[0] is NULL:
            cfronts_new[0] = NULL
        else:
            cfronts_new[0] = cfronts_tmp[0]
            cfronts_tmp[0] = NULL
        categories_now_n[0] = categories_new_n[0]
        if categories_tmp_n[0] is NULL:
            categories_new_n[0] = NULL
        else:
            categories_new_n[0] = categories_tmp_n[0]
            categories_tmp_n[0] = NULL
        grid_now = grid_new
        if grid_tmp is NULL:
            grid_new = NULL
        else:
            grid_new = grid_tmp
            grid_tmp = NULL

    timesteps_ts["old"] = timesteps_ts["now"]
    timesteps_ts["now"] = timesteps_ts["new"]
    timesteps_ts["new"] = None


def import_fronts(infile, feature_name, timestep, conf_grid):
    _r = read_feature_file(infile, feature_name=feature_name, timestep=timestep)
    fronts = _r["features"]
    jdat = _r["jdat"]
    name_info = feature_name.replace("features_", "info_")
    info = jdat[name_info]
    # SR_TMP <
    if not any(f.neighbors for f in fronts):
        log.info(f"find neighbors of {len(fronts)} '{feature_name}' features")
        const = default_constants(
            nx=conf_grid["nx"],
            ny=conf_grid["ny"],
            connectivity=conf_grid["connectivity"],
        )
        features_find_neighbors(fronts, const)
    # SR_TMP >
    return fronts, info


cdef void _run_front_surgery_core(
    dict timesteps_ts,
    list fronts_raw,
    cRegions* cfronts_new,
    cRegions* cfronts_now,
    cRegions* cfronts_old,
    int n_categories,
    int* categories_new_n,
    int* categories_now_n,
    int* categories_old_n,
    cGrid* grid_new,
    cGrid* grid_now,
    cGrid* grid_old,
    cConstants* constants,
    dict conf,
) except *:
    cdef bint debug = False
    cdef list thresholds = conf["thresholds"]
    cdef list fronts_grouped
    cdef dict fronts_cat
    cdef int n_features
    cdef bint create_in_features = False
    cdef bint ignore_missing_neighbors = False
    cdef cRegions cfronts_raw

    global CATEGORIES # SR_TMP

    # Deal with new fronts (always except last two timesteps)
    if fronts_raw is not None:
        # SR_TMP <
        # Convert features to cregions
        if debug: log.debug("<<< fronts_raw -> cfronts_raw")
        n_features = len(fronts_raw)
        cfronts_raw = cregions_create(n_features)
        features_to_cregions(
            fronts_raw,
            n_features,
            &cfronts_raw,
            cregion_conf_default(),
            ignore_missing_neighbors,
            grid_new,
            constants,
        )
        # SR_TMP >

        # Categorize fronts by size
        categorize_fronts(
            &cfronts_raw,
            cfronts_new,
            n_categories,
            categories_new_n,
            thresholds,
        )

        # DBG_BLOCK <
        if debug:
            log.debug("\n++++ NEW ++++")
            n = 0
            for i in range(n_categories):
                log.debug(f"{CATEGORIES[i]:10}{cfronts_new[i].n:3}")
            n += cfronts_new[i].n
            log.debug("-"*14)
            log.debug(f"{'sum':10}{n:3}")
            log.debug("="*14)
            log.debug("")
        # DBG_BLOCK >

        # Run first front surgery step in isolation (spatial and temporal)
        # Get rid of obvious clutter and re-merge features inside categories
        # (Isolation means only a single level at a single timestep)
        _front_surgery_isolated_core(
            cfronts_new,
            thresholds,
            n_categories,
            categories_new_n,
            grid_new,
            constants,
        )

    # DBG_BLOCK <
    if debug:
        if cfronts_new is not NULL:
            log.debug("\n++++ NEW ++++")
            n = 0
            for i in range(n_categories):
                log.debug(f"{CATEGORIES[i]:10}{cfronts_new[i].n:3}")
                n += cfronts_new[i].n
            log.debug("-"*14)
            log.debug(f"{'sum':10}{n:3}")
            log.debug("="*14)
        if cfronts_now is not NULL:
            log.debug("\n++++ NOW ++++")
            n = 0
            for i in range(n_categories):
                log.debug(f"{CATEGORIES[i]:10}{cfronts_now[i].n:3}")
                n += cfronts_now[i].n
            log.debug("-"*14)
            log.debug(f"{'sum':10}{n:3}")
            log.debug("="*14)
        if cfronts_old is not NULL:
            log.debug("\n++++ OLD ++++")
            n = 0
            for i in range(n_categories):
                log.debug(f"{CATEGORIES[i]:10}{cfronts_old[i].n:3}")
                n += cfronts_old[i].n
            log.debug("-"*14)
            log.debug(f"{'sum':10}{n:3}")
            log.debug("="*14)
        log.debug("")
    # DBG_BLOCK >

    if cfronts_now is not NULL:
        # Run second front surgery step across timesteps
        # Check for overlap of potential clutter with non-clutter
        # at previous or subsequent timestep (which are equivalent)
        # (Note that this is applied to the NEW features, not the NOWs
        # to which the first surgery step has just been applied before)
        _front_surgery_temporal_core(
            cfronts_new,
            cfronts_now,
            cfronts_old,
            n_categories,
            categories_new_n,
            categories_now_n,
            categories_old_n,
            grid_new,
            grid_now,
            grid_old,
            constants,
            conf,
        )

    # DBG_BLOCK <
    if debug:
        if cfronts_new is not NULL:
            log.debug("\n++++ NEW ++++")
            n = 0
            for i in range(n_categories):
                log.debug(f"{CATEGORIES[i]:10}{cfronts_new[i].n:3}")
                n += cfronts_new[i].n
            log.debug("-"*14)
            log.debug(f"{'sum':10}{n:3}")
            log.debug("="*14)
        if cfronts_now is not NULL:
            log.debug("\n++++ NOW ++++")
            n = 0
            for i in range(n_categories):
                log.debug(f"{CATEGORIES[i]:10}{cfronts_now[i].n:3}")
                n += cfronts_now[i].n
            log.debug("-"*14)
            log.debug(f"{'sum':10}{n:3}")
            log.debug("="*14)
        if cfronts_old is not NULL:
            log.debug("\n++++ OLD ++++")
            n = 0
            for i in range(n_categories):
                log.debug(f"{CATEGORIES[i]:10}{cfronts_old[i].n:3}")
                n += cfronts_old[i].n
            log.debug("-"*13)
            log.debug("{'sum':10}{n:3}")
            log.debug("="*14)
        log.debug("")
    # DBG_BLOCK >


cpdef dict front_surgery_isolated(
    list fronts,
    list thresholds,
    int nx,
    int ny,
    int connectivity=8,
    int n_neighbors_max=8,
):
    """Run the first major step of front surgery on an isolated field.

    Input:
    A list of front subfeatures which have already been internally split,
    i.e. not the raw output of the initial front identification tool.
    The features are internally split, i.e. can touch each other.

    Output:
    Multiple categorized lists of front features (clutter and non-clutter).
    Features in the same category have been merged, i.e. the remaining
    distinct features don't touch each other. However, features in different
    categories may touch each other.

    """
    # Set up constants
    cdef cConstants constants = cConstants(
        nx=nx, ny=ny, connectivity=connectivity, n_neighbors_max=n_neighbors_max,
    )

    # Initialize grid
    cdef np.ndarray[np.float32_t, ndim=2] field = np.zeros([nx, ny], dtype=np.float32)
    cdef cGrid grid = grid_create(field, constants)
    pixel_region_table_alloc_grid(&grid.pixel_region_table, &constants)

    # Allocate/initialize lookup tables
    pixel_status_table_alloc(&grid.pixel_status_table, &constants)
    neighbor_link_stat_table_alloc(&grid.neighbor_link_stat_table, &constants)

    # SR_TODO pack all the usual conversion stuff from python to cython in
    # SR_TODO separate functions (conversion/back-conversion) to reduce overhead!

    # Turn features into cregions
    cdef int n_fronts = len(fronts)
    cdef cRegions cfronts_raw = cregions_create(n_fronts)
    cdef bint ignore_missing_neighbors = False
    features_to_cregions(
        fronts,
        n_fronts,
        &cfronts_raw,
        cregion_conf_default(),
        ignore_missing_neighbors,
        &grid,
        &constants,
    )

    # Initialize category size array
    global N_CATEGORIES
    cdef int n_categories = N_CATEGORIES
    cdef int* categories_n = <int*>malloc(n_categories*sizeof(int))
    for i in range(n_categories):
        categories_n[i] = 0

    # Categorize fronts by size
    cdef cRegions* cfronts = <cRegions*>malloc(n_categories*sizeof(cRegions))
    for i in range(n_categories):
        cfronts[i] = cregions_create(10)
    categorize_fronts(&cfronts_raw, cfronts, n_categories, categories_n, thresholds)

    # Run surgery
    _front_surgery_isolated_core(
        cfronts, thresholds, n_categories, categories_n, &grid, &constants,
    )

    # Turn cregion into Feature objects
    cdef dict fronts_new = cfronts2fronts(
        cfronts, n_categories, categories_n, 0, 0, &grid, &constants,
    )

    # Clean up cregions etc.
    cregions_cleanup(&cfronts_raw, cleanup_regions=True)
    free(categories_n)

    # Clean up grid
    grid_cleanup(&grid)
    features_reset_cregion(fronts, warn=False)
    cdef list fronts_
    for fronts_ in fronts_new.values():
        features_reset_cregion(fronts_, warn=False)

    return fronts_new


cdef void _front_surgery_isolated_core(
    cRegions* cfronts_cat,
    list thresholds,
    int n_categories,
    int* categories_n,
    cGrid* grid,
    cConstants* constants,
):
    # Sort out the most obvious clutter (smallest features)
    identify_obvious_clutter(
        cfronts_cat, n_categories, categories_n, thresholds, constants,
    )

    # Re-merge adjacent fronts which belong to the same size category
    # SR_TODO is reset_connections necessary?
    merge_features_in_categories(
        cfronts_cat,
        n_categories,
        categories_n,
        grid,
        constants,
        reset_connections=True,
    )

    # Merge small into medium category
    merge_categories(
        cfronts_cat,
        category_small,
        category_medium,
        n_categories,
        categories_n,
        constants,
        neighbors_across_categories=True,
    )

    # Categorize medium features with clutter neighbors as clutter
    identify_clutter_neighbors(
        cfronts_cat, category_medium, category_clutter1, n_categories, categories_n,
    )

    # Merge medium into large category
    merge_categories(
        cfronts_cat,
        category_medium,
        category_large,
        n_categories,
        categories_n,
        constants,
        neighbors_across_categories=False,
    )

    # Re-merge adjacent fronts which belong to the same size category
    merge_features_in_categories(
        cfronts_cat,
        n_categories,
        categories_n,
        grid,
        constants,
        reset_connections=False,
    )

    # Move the largest clutter clusters (definitely clutter)
    # into a separate category (from cluster1 into cluster0)
    separate_big_clutter_clusters(cfronts_cat, n_categories, categories_n, thresholds)


cdef void merge_categories(
    cRegions* cfronts_cat,
    int i_source,
    int i_target,
    int n_categories,
    int* categories_n,
    cConstants* constants,
    bint neighbors_across_categories,
):
    cdef int i

    # All all source features to the target category
    for i in range(cfronts_cat[i_source].n):
        cregions_link_region(
                &cfronts_cat[i_target],
                cfronts_cat[i_source].regions[i],
                cleanup = False,
                unlink_pixels = False,
            )
        categories_n[i_target] += 1
    cfronts_cat[i_source].n = 0
    categories_n[i_source] = 0

    # Determine neighbors (i.e. connected regions)
    cdef bint reset_existing = True
    if neighbors_across_categories:

        # Consider all features, regardless of category
        find_neighbors_across_categories(
                cfronts_cat,
                n_categories,
                categories_n,
                constants,
            )
    else:
        # Only consider features in the same category
        for i in range(n_categories):
            cregions_find_connected(&cfronts_cat[i], reset_existing, constants)


cdef void identify_obvious_clutter(
    cRegions* cfronts_cat,
    int n_categories,
    int* categories_n,
    list thresholds,
    cConstants* constants,
):
    cdef int i
    cdef int j
    cdef int k
    cdef int i_clutter0 = category_clutter0
    cdef int i_clutter1 = category_clutter1
    cdef int i_small = category_small
    cdef int i_large = category_large
    cdef int n_small_old = cfronts_cat[i_small].n
    cdef cRegion** cfronts_tmp = <cRegion**>malloc(n_small_old*sizeof(cRegion*))
    for i in range(n_small_old):
        cfronts_tmp[i] = cfronts_cat[i_small].regions[i]
    cdef int n_clutter0 = 0
    cdef int n_clutter1 = 0
    cdef cRegion* cfront
    cdef cRegion* neighbor
    for i in range(n_small_old):
        cfront = cfronts_tmp[i]

        # Check if there are no neighbors
        if cfront.connected_n == 0:
            cregions_link_region(
                &cfronts_cat[i_clutter0], cfront, cleanup=False, unlink_pixels=False,
            )
            n_clutter0 += 1
            cfronts_tmp[i] = NULL
            continue

        # Check if any neighbor is large
        for j in range(cfront.connected_n):
            neighbor = cfront.connected[j]
            for k in range(cfronts_cat[i_large].n):
                if cfronts_cat[i_large].regions[k].id == neighbor.id:
                    break
            else:
                continue
            break
        else:
            # No neighbor is large -> move to clutter
            cregions_link_region(
                &cfronts_cat[i_clutter1], cfront, cleanup=False, unlink_pixels=False,
            )
            n_clutter1 += 1
            cfronts_tmp[i] = NULL

    log.info(
        f"turned {n_clutter0}/{n_small_old} small features into clutter0 (no neighbors)"
    )
    log.info(
        f"turned {n_clutter1}/{n_small_old} small features into clutter1 (no large "
        f"neighbors)"
    )

    # Move remaining small features back into category
    j = 0
    for i in range(n_small_old):
        if cfronts_tmp[i] is not NULL:
            cfronts_cat[i_small].regions[j] = cfronts_tmp[i]
            j += 1
    cfronts_cat[i_small].n = j
    free(cfronts_tmp)

    # Update category sizes
    for i in range(n_categories):
        categories_n[i] = cfronts_cat[i].n

    # Re-determine neighbors (aka connected regions)
    cdef bint reset_existing = True
    for i in range(n_categories):
        cregions_find_connected(&cfronts_cat[i], reset_existing, constants)


cdef int categorize_fronts(
    cRegions* cfronts_raw,
    cRegions* cfronts_cat,
    int n_categories,
    int* categories_n,
    list thresholds,
) except -1:
    cdef str _name_ = "categorize_fronts"
    cdef int i
    cdef int j
    cdef int n

    # Check thresholds
    if len(thresholds) != 2:
        log.error(f"{_name_}: wrong number of thresholds ({len(thresholds)} != 2)")
        exit(4)
    cdef np.uint32_t lower_threshold = thresholds[0]
    cdef np.uint32_t upper_threshold = thresholds[1]

    # Initizlize category indices
    cdef int i_clutter0 = category_clutter0
    cdef int i_clutter1 = category_clutter1
    cdef int i_small = category_small
    cdef int i_medium = category_medium
    cdef int i_large = category_large

    # Sort fronts into categories
    cdef cRegion* cfront
    cdef cRegions* target
    for i in range(cfronts_raw.n):
        cfront = cfronts_raw.regions[i]
        if cfront.pixels_n < lower_threshold:
            target = &cfronts_cat[i_small]
            categories_n[i_small] += 1
        elif lower_threshold <= cfront.pixels_n < upper_threshold:
            target = &cfronts_cat[i_medium]
            categories_n[i_medium] += 1
        elif upper_threshold <= cfront.pixels_n:
            target = &cfronts_cat[i_large]
            categories_n[i_large] += 1
        cregions_link_region(target, cfront, cleanup=False, unlink_pixels=False)
    log.info(
        f"categorized {cfronts_raw.n} fronts: {categories_n[i_small]} small, "
        f"{categories_n[i_medium]} medium, {categories_n[i_large]} large"
    )
    return n_categories


cdef void merge_features_in_categories(
    cRegions* cfronts_cat,
    int n_categories,
    int* categories_n,
    cGrid* grid,
    cConstants* constants,
    bint reset_connections,
):
    cdef int i
    cdef int j
    cdef int k
    cdef int n
    global CATEGORIES

    # Note: At this point, there should be no connections between regions
    # that belong to different categories, otherwise features are merged
    # across category boundaries.
    cdef bint reset_existing = True
    if reset_connections:
        # Re-determine all neighbors in categories
        for i in range(n_categories):
            cregions_find_connected(&cfronts_cat[i], reset_existing, constants)

    # Merge adjacent features inside categories
    cdef bint exclude_seed_points = False
    cdef int nold
    cdef int nnew
    cdef cRegions

    for i in range(n_categories):
        nold = cfronts_cat[i].n
        cregions_merge_connected_inplace(
            &cfronts_cat[i],
            grid,
            exclude_seed_points,
            nold,
            cregion_conf_default(),
            constants,
        )
        nnew = cfronts_cat[i].n
        if nnew < nold:
            log.info(f"merged '{CATEGORIES[i]}' features: {nold} -> {nnew}")
            categories_n[i] = nnew

    # SR_TODO Test whether this is actually necessary (I think it shouldn't be)!
    # Re-compute fresh boundaries of categories of size 1
    for i in range(n_categories):
        if cfronts_cat[i].n == 1:
            cregion_determine_boundaries(cfronts_cat[i].regions[0], grid)

    # Connect regions across categories
    find_neighbors_across_categories(cfronts_cat, n_categories, categories_n, constants)


cdef void find_neighbors_across_categories(
    cRegions* cfronts_cat, int n_categories, int* categories_n, cConstants* constants,
):
    # Find neighbors across regions
    cdef cRegions cfronts_all = flatten_cfronts_arr(cfronts_cat, n_categories)
    cdef bint reset_existing = True
    cregions_find_connected(&cfronts_all, reset_existing, constants)
    # Put regions back into categories (separate cregions)
    unflatten_cfronts_all(&cfronts_all, n_categories, categories_n, cfronts_cat)
    cregions_cleanup(&cfronts_all, cleanup_regions=False)


# SR_TMP <
cdef void features_cat_to_cregions_arr(
    dict fronts_cat,
    cRegions* cfronts_cat,
    int n_categories,
    int* categories_n,
    cGrid* grid,
    cConstants* constants,
) except *:
    cdef str _name_ = "features_cat_to_cregions_arr"
    cdef bint debug = False
    if debug:
        log.debug(f"< {_name_}")
    cdef int i
    cdef int n_features
    cdef bint ignore_missing_neighbors = True
    global CATEGORIES
    cdef list categories = CATEGORIES
    cdef int n_features_tot = 0
    for i in range(n_categories):
        features = fronts_cat[categories[i]]
        n_features = len(features)
        n_features_tot += n_features
        cfronts_cat[i] = cregions_create(n_features)
        features_to_cregions(
            features,
            n_features,
            &cfronts_cat[i],
            cregion_conf_default(),
            ignore_missing_neighbors,
            grid,
            constants,
        )
        categories_n[i] = cfronts_cat[i].n
    find_neighbors_across_categories(cfronts_cat, n_categories, categories_n, constants)
    if debug:
        log.debug(f"> {_name_}")


cdef dict cfronts2fronts(
    cRegions* cfronts_cat,
    int n_categories,
    int* categories_n,
    np.uint64_t base_id,
    np.uint64_t timestep,
    cGrid* grid,
    cConstants* constants,
):
    # log.info("< cfronts2fronts")
    cdef dict fronts_cat = {}
    cdef int i
    cdef int j
    cdef int k
    cdef cRegions cfronts_all = flatten_cfronts_arr(cfronts_cat, n_categories)
    cdef bint ignore_missing_neighbors = False
    cdef list merged_features = cregions_create_features(
        &cfronts_all, base_id, ignore_missing_neighbors, grid, constants,
    )

    # SR_TMP < SR_TODO Move into cregions_create_features (timestep should ALWAYS be set)
    cdef Feature front
    for front in merged_features:
        front.timestep = timestep
    # SR_TMP >

    cregions_cleanup(&cfronts_all, cleanup_regions=True)

    # SR_TMP >
    for i in range(n_categories):
        cregions_cleanup(&cfronts_cat[i], cleanup_regions=False)
    free(cfronts_cat)
    cdef str category
    global CATEGORIES
    cdef list categories = CATEGORIES
    k = 0
    for i in range(n_categories):
        category = categories[i]
        fronts_cat[category] = []
        for j in range(categories_n[i]):
            fronts_cat[category].append(merged_features[k])
            k += 1
    # log.info("> cfronts2fronts")
    return fronts_cat


cdef cRegions flatten_cfronts_arr(cRegions* cfronts_cat, int n_categories):
    # log.info("< flatten_cfronts_arr")
    cdef int i
    cdef int j
    cdef int n_fronts = 0
    for i in range(n_categories):
        n_fronts += cfronts_cat[i].n
    cdef cRegions cfronts_all = cregions_create(n_fronts)
    cfronts_all.n = 0
    for i in range(n_categories):
        for j in range(cfronts_cat[i].n):
            cregions_link_region(
                &cfronts_all,
                cfronts_cat[i].regions[j],
                cleanup=False,
                unlink_pixels=False,
            )
    # log.info("> flatten_cfronts_arr")
    return cfronts_all


cdef void unflatten_cfronts_all(
    cRegions* cfronts_all, int n_categories, int* categories_n, cRegions* cfronts_cat,
):
    # log.info("< unflatten_cfronts_all")
    cdef int i
    cdef int j
    cdef int k
    cdef int n
    cdef cRegions* cfronts
    k = 0
    for i in range(n_categories):
        cfronts_cat[i].n = 0 # 'soft reset'
        for j in range(categories_n[i]):
            if k > cfronts_all.n:
                log.error(
                    f"unflatten_cfronts_all: k > cfronts_all.n: {k} > {cfronts_all.n}"
                )
                exit(66)
            cregions_link_region(
                &cfronts_cat[i],
                cfronts_all.regions[k],
                cleanup=False,
                unlink_pixels=False,
            )
            k += 1
    # log.info("> unflatten_cfronts_all")
# SR_TMP >


cdef tuple dict2lists(dict dict_):
    cdef str key
    cdef list keys = []
    cdef list vals = []
    cdef list val
    for key, val in sorted(dict_.items()):
        keys.append(key)
        vals.append([v for v in val])
    return keys, vals


cdef void identify_clutter_neighbors(
    cRegions* cfronts_cat,
    int i_source,
    int i_target,
    int n_categories,
    int* categories_n,
) except *:
    cdef int i
    cdef int j
    cdef int k
    cdef int n
    cdef int nold
    cdef int nnew

    # Initialize temporary array to _target regions
    cdef int n_clutter = cfronts_cat[i_target].n
    cdef cRegion** clutter_tmp = <cRegion**>malloc(n_clutter*sizeof(cRegion*))
    for i in range(n_clutter):
        clutter_tmp[i] = cfronts_cat[i_target].regions[i]

    # Initialize temporary array to source regions
    cdef int n_source = cfronts_cat[i_source].n
    cdef cRegion** small_tmp = <cRegion**>malloc(n_source*sizeof(cRegion*))
    for i in range(n_source):
        small_tmp[i] = cfronts_cat[i_source].regions[i]

    nold = cfronts_cat[i_source].n
    nnew = cfronts_cat[i_source].n
    cdef cRegion* cfront_source
    cdef cRegion* cfront_connected
    for i in range(n_source):
        cfront_source = cfronts_cat[i_source].regions[i]

        # Check whether any neighbors are clutter
        for j in range(cfront_source.connected_n):
            cfront_connected = cfronts_cat[i_source].regions[i].connected[j]

            # Check clutter for neighbor
            for k in range(n_clutter):
                if clutter_tmp[k].id == cfront_connected.id:
                    # Neighbor is clutter
                    break
            else:
                # Neighbor is not clutter
                continue
            break
        else:
            continue
        nnew -= 1

        # Remove from source
        small_tmp[i] = NULL
        categories_n[i_source] -= 1
        cfronts_cat[i_source].n -= 1

        # Add to target
        cregions_link_region(
            &cfronts_cat[i_target],
            cfront_source,
            cleanup=False,
            unlink_pixels=False,
        )
        categories_n[i_target] += 1

    # Move remaining source regions back into resp. cregion
    n = 0
    for i in range(n_source):
        if small_tmp[i] is not NULL:
            cfronts_cat[i_source].regions[n] = small_tmp[i]
            n += 1
    cfronts_cat[i_source].n = n

    global CATEGORIES
    log.info(
        f"discard {nold - nnew}/{nold} {CATEGORIES[i_source]} front clusters with no "
        f"bigger neighbors (mark "
    )


cdef separate_big_clutter_clusters(
    cRegions* cfronts_cat, int n_categories, int* categories_n, list thresholds,
):
    cdef int i
    cdef int j
    cdef int threshold = thresholds[1]
    cdef int i_clutter0 = category_clutter0
    cdef int i_clutter1 = category_clutter1
    cdef int n_clut1_old = cfronts_cat[i_clutter1].n
    cdef cRegion** cfronts_tmp = <cRegion**>malloc(n_clut1_old*sizeof(cRegion*))
    for i in range(n_clut1_old):
        cfronts_tmp[i] = cfronts_cat[i_clutter1].regions[i]

    cdef cRegion* cfront
    for i in range(n_clut1_old):
        cfront = cfronts_tmp[i]
        if cfront.pixels_n >= threshold:
            cregions_link_region(
                &cfronts_cat[i_clutter0], cfront, cleanup=False, unlink_pixels=False,
            )
            categories_n[i_clutter0] += 1
            cfronts_tmp[i] = NULL
            categories_n[i_clutter1] -= 1
    j = 0
    for i in range(n_clut1_old):
        if cfronts_tmp[i] is not NULL:
            cfronts_cat[i_clutter1].regions[j] = cfronts_tmp[i]
            j += 1
    cfronts_cat[i_clutter1].n = j
    free(cfronts_tmp)
    cdef int n_clut1_new = cfronts_cat[i_clutter1].n
    log.info(
        f"separated biggest {n_clut1_old - n_clut1_new}/{n_clut1_old} clutter clusters "
        f"as definite clutter (n >= {thresholds[1]})"
    )


cpdef dict front_surgery_temporal(
    dict fronts_new,
    dict fronts_now,
    dict fronts_old,
    int nx,
    int ny,
    int max_size = -1,
    float min_overlap = 0.8,
    int connectivity = 8,
    int n_neighbors_max = 8,
    bint merge_nonclutter = False,
):
    """Run temporal front surgery step.

    Algorithm:
     * Recover all <src> features at NOW with all large features at <ts>:
        * If the feature exceeds the max. size, skip it (optimization step)
        * Compute total overlap with all large features at <ts>
        * Compute the relative overlap wrt. the size of the feature
        * Recover the front if the relative overlap is sufficient
     * <src> refers to "clutter1" and "clutter0" (in that order)
     * <ts> refers to OLD and/or NEW (depending on what's available)
        * Finally, merge all adjacent features in the same categories

    """
    if fronts_new is None and fronts_old is None:
        err = "Either fronts_new or fronts_old must not be None"
        raise ValueError(err)

    # Set up constants
    cdef cConstants constants = cConstants(
        nx=nx, ny=ny, connectivity=connectivity, n_neighbors_max=n_neighbors_max,
    )

    # Initialize grids
    cdef np.ndarray[np.float32_t, ndim=2] field = np.zeros([nx, ny], dtype=np.float32)
    cdef cGrid grid_new = grid_create_empty(constants)
    cdef cGrid grid_now = grid_create_empty(constants)
    cdef cGrid grid_old = grid_create_empty(constants)
    grid_create_pixels(&grid_now, field)
    if fronts_new is not None:
        grid_create_pixels(&grid_new, field)
    if fronts_old is not None:
        grid_create_pixels(&grid_old, field)
    pixel_region_table_alloc_grid(&grid_now.pixel_region_table, &constants)

    # Allocate/initialize lookup tables
    pixel_status_table_alloc(&grid_new.pixel_status_table, &constants)
    pixel_status_table_alloc(&grid_now.pixel_status_table, &constants)
    pixel_status_table_alloc(&grid_old.pixel_status_table, &constants)
    neighbor_link_stat_table_alloc(&grid_new.neighbor_link_stat_table, &constants)
    neighbor_link_stat_table_alloc(&grid_now.neighbor_link_stat_table, &constants)
    neighbor_link_stat_table_alloc(&grid_old.neighbor_link_stat_table, &constants)

    # Initialize category size arrays
    global N_CATEGORIES
    cdef int n_categories = N_CATEGORIES
    cdef int* categories_now_n = <int*>malloc(n_categories*sizeof(int))
    cdef int* categories_new_n = NULL
    cdef int* categories_old_n = NULL
    if fronts_new is not None:
        categories_new_n = <int*>malloc(n_categories*sizeof(int))
    if fronts_old is not None:
        categories_old_n = <int*>malloc(n_categories*sizeof(int))
    for i in range(n_categories):
        categories_now_n[i] = 0
        if categories_new_n is not NULL:
            categories_new_n[i] = 0
        if categories_old_n is not NULL:
            categories_old_n[i] = 0

    # Turn features into cregions
    cdef int n = n_categories
    cdef cRegions* cfronts_now = <cRegions*>malloc(n*sizeof(cRegions))
    cdef cRegions* cfronts_new = NULL
    cdef cRegions* cfronts_old = NULL
    features_cat_to_cregions_arr(
        fronts_now, cfronts_now, n_categories, categories_now_n, &grid_now, &constants,
    )
    if fronts_new is not None:
        cfronts_new = <cRegions*>malloc(n*sizeof(cRegions))
        features_cat_to_cregions_arr(
            fronts_new,
            cfronts_new,
            n_categories,
            categories_new_n,
            &grid_new,
            &constants,
        )
    if fronts_old is not None:
        cfronts_old = <cRegions*>malloc(n*sizeof(cRegions))
        features_cat_to_cregions_arr(
            fronts_old,
            cfronts_old,
            n_categories,
            categories_old_n,
            &grid_old,
            &constants,
        )

    # Run surgery
    cdef dict conf = {"max_size_temporal_step": max_size, "min_overlap": min_overlap}
    _front_surgery_temporal_core(
        cfronts_new,
        cfronts_now,
        cfronts_old,
        n_categories,
        categories_new_n,
        categories_now_n,
        categories_old_n,
        &grid_new,
        &grid_now,
        &grid_old,
        &constants,
        conf,
    )

    # Merge all non-clutter features
    # Note: In the real surgery, this is only done later after the NOW
    # features have themselves acted as OLD features, otherwise the surgery
    # becomes dependent on the start timestep.
    if merge_nonclutter:
        merge_categories(
            cfronts_now,
            category_medium,
            category_large,
            n_categories,
            categories_now_n,
            &constants,
            neighbors_across_categories=False,
        )
        merge_features_in_categories(
            cfronts_now,
            n_categories,
            categories_now_n,
            &grid_now,
            &constants,
            reset_connections=False,
        )

    # Turn cregion into Feature objects (NOW only, others unchaged)
    cdef dict fronts_now_out = cfronts2fronts(
        cfronts_now, n_categories, categories_now_n, 0, 0, &grid_now, &constants,
    )

    # Reset cregions of features
    cdef dict fronts_dict
    cdef list fronts
    cdef Feature front
    for fronts_dict in (fronts_new, fronts_now, fronts_old, fronts_now_out):
        if fronts_dict is not None:
            for fronts in fronts_dict.values():
                for front in fronts:
                    front.reset_cregion(warn=False)

    # Clean up category size arrays
    free(categories_now_n)
    if categories_new_n is not NULL:
        free(categories_new_n)
    if categories_old_n is not NULL:
        free(categories_old_n)

    # Clean up grids
    grid_cleanup(&grid_new)
    grid_cleanup(&grid_now)
    grid_cleanup(&grid_old)
    if fronts_new is not None:
        for fronts in fronts_new.values():
            features_reset_cregion(fronts, warn=False)
    if fronts_now is not None:
        for fronts in fronts_now.values():
            features_reset_cregion(fronts, warn=False)
    if fronts_old is not None:
        for fronts in fronts_old.values():
            features_reset_cregion(fronts, warn=False)
    for fronts in fronts_now_out.values():
        features_reset_cregion(fronts, warn=False)

    return fronts_now_out


cdef void _front_surgery_temporal_core(
    cRegions* cfronts_new,
    cRegions* cfronts_now,
    cRegions* cfronts_old,
    int n_categories,
    int* categories_new_n,
    int* categories_now_n,
    int* categories_old_n,
    cGrid* grid_new,
    cGrid* grid_now,
    cGrid* grid_old,
    cConstants* constants,
    dict conf,
) except *:
    cdef bint debug = False
    log.info("\ntemporal front surgery step")
    cdef int i_source
    cdef int i_target
    cdef int i_large = category_large
    cdef str info

    # Only check the post-re-merge clutter for overlap
    # Cluster categories 1 and 2 are definitely clutter

    # Old/clutter1
    if cfronts_old is not NULL:
        if debug: log.debug("compare features with those at previous timestep")
        i_source = category_clutter1
        i_target = category_large
        info = "now <-> old"
        _front_surgery_temporal_core_core(
            info,
            i_source,
            i_target,
            cfronts_now,
            cfronts_old,
            n_categories,
            categories_now_n,
            categories_old_n,
            grid_now,
            grid_old,
            constants,
            conf,
        )

    # New/clutter1
    if cfronts_new is not NULL:
        if debug: log.debug("compare features with those at subsequent timestep")
        i_source = category_clutter1
        i_target = category_medium
        info = "now <-> new"
        _front_surgery_temporal_core_core(
            info,
            i_source,
            i_target,
            cfronts_now,
            cfronts_new,
            n_categories,
            categories_now_n,
            categories_new_n,
            grid_now,
            grid_new,
            constants,
            conf,
        )

    # Old/clutter0
    if cfronts_old is not NULL:
        if debug: log.debug("compare features with those at previous timestep")
        i_source = category_clutter0
        i_target = category_medium
        info = "now <-> old"
        _front_surgery_temporal_core_core(
            info,
            i_source,
            i_target,
            cfronts_now,
            cfronts_old,
            n_categories,
            categories_now_n,
            categories_old_n,
            grid_now,
            grid_old,
            constants,
            conf,
        )

    # New/clutter0
    if cfronts_new is not NULL:
        if debug: log.debug("compare features with those at subsequent timestep")
        i_source = category_clutter0
        i_target = category_medium
        info = "now <-> new"
        _front_surgery_temporal_core_core(
            info,
            i_source,
            i_target,
            cfronts_now,
            cfronts_new,
            n_categories,
            categories_now_n,
            categories_new_n,
            grid_now,
            grid_new,
            constants,
            conf,
        )

    # Re-merge adjacent fronts which belong to the same size category
    merge_features_in_categories(
        cfronts_now,
        n_categories,
        categories_now_n,
        grid_now,
        constants,
        reset_connections=True
    )


cdef void _front_surgery_temporal_core_core(
    str info,
    int i_source,
    int i_target,
    cRegions* cfronts_now,
    cRegions* cfronts_other_cat,
    int n_categories,
    int* categories_now_n,
    int* categories_other_n,
    cGrid* grid_now,
    cGrid* grid_other,
    cConstants* constants,
    dict conf,
) except *:
    cdef bint debug = False
    cdef bint  timing = False
    cdef int i
    cdef int j
    cdef int k
    cdef int n
    cdef int i_overlap
    cdef np.float32_t max_size = conf["max_size_temporal_step"]
    cdef np.float32_t rel_overlap
    cdef np.float32_t min_overlap = conf["min_overlap"]

    # Prepare logging
    cdef str logfile = conf.get("logfile_temporal_step")
    cdef bint logging
    if not logfile:
        logging = False
    else:
        logging = True
    cdef object flog
    if logging:
        flog = open(logfile, "a")

    # Check clutter for overlap with large features at other timestep
    cdef int i_large = category_large
    cdef cRegions* cfronts_other = &cfronts_other_cat[i_large]
    cdef cRegions* cfronts_target = &cfronts_now[i_target]
    cdef cRegions* cfronts_source = &cfronts_now[i_source]
    cdef cRegion* cfront
    cdef int nold = cfronts_source.n
    cdef cRegion** cfronts_tmp = <cRegion**>malloc(nold*sizeof(cRegion*))
    for i in range(nold):
        cfronts_tmp[i] = cfronts_source.regions[i]
    for i in range(nold):
        cfront = cfronts_tmp[i]

        # Check size
        if max_size > 0 and cfront.pixels_n > max_size:
            if debug:
                log.debug(
                    f" - skip huge feature {cfront.id} ({cfront.pixels_n} > {max_size})"
                )
            continue

        # Start measuring
        if debug or logging:
            t0 = time.clock()

        # Perform computations
        n_overlap = 0
        for j in range(cfronts_other.n):
            n_overlap += cregion_overlap_n(cfront, cfronts_other.regions[j])
        rel_overlap = (<float>n_overlap)/(<float>cfront.pixels_n)

        # Stop measuring, write log file
        if (debug and timing) or logging:
            dt = time.clock() - t0
            log.info(
                f" [{cfront.id:5}] {cfront.pixels_n:9} pixels; {n_overlap:9} pixels "
                f"overlap; {rel_overlap:3.2f} % overlap; {dt:4.3f} sec"
            )
        elif debug and not timing:
            log.info(
                f" [{cfront.id:5}] {cfront.pixels_n:9} pixels; {n_overlap:9} pixels "
                f"overlap; {rel_overlap:3.2f} % overlap"
            )
        if logging:
            flog.write(f"{cfront.pixels_n} {n_overlap} {dt}\n")

        # Recover front if overlap sufficient
        if rel_overlap >= min_overlap:
            if debug:
                log.debug(
                    f"   -> front {cfront.id} not clutter (overlap {rel_overlap:2.1%})"
                )
            cfronts_tmp[i] = NULL
            cregions_link_region(
                    cfronts_target,
                    cfront,
                    cleanup = False,
                    unlink_pixels = False,
                )
    j = 0
    for i in range(nold):
        if cfronts_tmp[i] is not NULL:
            cfronts_source.regions[j] = cfronts_tmp[i]
            j += 1
    cfronts_source.n = j
    cdef int nnew = cfronts_source.n

    # Update category sizes
    categories_now_n[i_source] = cfronts_source.n
    categories_now_n[i_target] = cfronts_target.n

    global CATEGORIES
    log.info(f"{info}: restored {nold - nnew}/{nold} {CATEGORIES[i_source]} features")

    if logging:
        flog.close()
