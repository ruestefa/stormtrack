# !/usr/bin/env python3

from __future__ import print_function

# C: Standard library
from cpython.object cimport Py_EQ
from cpython.object cimport Py_GE
from cpython.object cimport Py_GT
from cpython.object cimport Py_LE
from cpython.object cimport Py_LT
from cpython.object cimport Py_NE

# C: C libraries
from libc.math cimport pow
from libc.math cimport sqrt
from libc.stdlib cimport RAND_MAX
from libc.stdlib cimport exit
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport rand
from libc.stdlib cimport srand

# C: Third-party
cimport cython
cimport numpy as np

# Standard library
import functools
import logging as log
import os
import sys
from copy import copy
from copy import deepcopy
from pprint import pprint
from pprint import pformat

# Third-party
import cython
import numpy as np
import PIL
import pyproj
import scipy as sp
import shapely.geometry as geo
import shapely.ops
from cython.parallel import prange

# Local
from .constants import default_constants
from ..utils.spatial import points_area_lonlat_reg
from ..utils.spatial import paths_lonlat_to_mask
try:
    from ..utils.various import NoIndent
except ImportError:
    print("warning: failed import: ..utils.various.NoIndent")
    def NoIndent(arg):
        return arg


# identify_features
# find_features_2d_threshold_seeded
# _find_features_threshold_random_seeds
# c_find_features_2d_threshold_seeds
# c_find_features_2d_threshold_seeds_core
# grow_cregion_rec
# init_random_seeds
# pop_random_unassigned_pixel
# random_int
# cregions_merge_connected_inplace
# cregions_merge_connected
# _cregions_merge_connected_core
# collect_pixels
# create_feature
# cregion_collect_connected_regions
# _cregion_collect_connected_regions_rec
# assign_cpixel
# find_features_2d_threshold
# eliminate_regions_by_size
# find_existing_region
# merge_adjacent_features
# feature_split_regiongrow
# features_grow
# cfeatures_grow_core
# _replace_feature_associations
# split_regiongrow_levels
# csplit_regiongrow_levels
# csplit_regiongrow_levels_core
# extract_subregions_level
# collect_adjacent_pixels
# csplit_regiongrow_core
# assert_no_unambiguously_assigned_pixels
# regiongrow_advance_boundary
# regiongrow_resolve_multi_assignments
# resolve_multi_assignment
# dbg_print_selected_regions
# resolve_multi_assignment_best_connected_region
# resolve_multi_assignment_biggest_region
# resolve_multi_assignment_strongest_region
# resolve_multi_assignment_mean_strongest_region
# cpixel_count_neighbors_in_cregion
# regiongrow_assign_pixel
# find_minima_2d
# find_maxima_2d
# c_find_extrema_2d
# _c_find_extrema_2d_core
# features_to_cregions
# dbg_features_check_unique_pixels
# dbg_check_features_cregion_pixels
# features_neighbors_to_cregions_connected
# feature_to_cregion
# pixels_find_boundaries
# Pixel
# Field2D
# Feature_rebuild
# Feature
# _feature__from_jdat__pixels_from_tables
# feature2d_from_jdat
# features_reset_cregion
# features_find_neighbors
# features_find_neighbors_core
# cregions_create_features
# cregions2features_connected2neighbors
# determine_shared_boundary_pixels
# initialize_surrounding_background_region
# _find_background_neighbor_pixels
# cregion_find_corresponding_feature
# cpixel2arr
# features_neighbors_id2obj
# features_neighbors_obj2id
# associate_features
# resolve_indirect_associations
# features_associates_obj2id
# features_associates_id2obj
# oldfeature_to_pixels
# cyclones_to_features


# Default type codes for feature id for common feature types
# fmt: off
DEFAULT_TYPE_CODES = dict(
    front850        = 110,
    warmfront850    = 111,
    coldfront850    = 112,
    front700        = 120,
    warmfront700    = 121,
    coldfront700    = 122,
    precip          = 300,
    prec01          = 310,
    prec02          = 311,
    prec1mmh        = 310,
    prec2mmh        = 311,
    prec5mmh        = 312,
    prec05mmh       = 315,
    prec01mmh       = 320,
    prec005mmh      = 325,
    prec001mmh      = 330,
    prec0005mmh     = 335,
    prec0001mmh     = 340,
    wind10m         = 410,
    wind700         = 420,
    wind850         = 430,
    cyclone         = 510,
    cyclone850      = 520,
    cyclone700      = 530,
    anticyclone     = 610,
    anticyclone850  = 620,
    anticyclone700  = 630,
)
# fmt: on


# :call: > --- callers ---
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::find_features_2d_threshold
# :call: v stormtrack::core::identification::split_regiongrow_levels
# :call: v stormtrack::core::grid::Grid
# :call: v stormtrack::core::constants::default_constants
def identify_features(
    fld,
    feature_name,
    nx,
    ny,
    lower,
    upper,
    minsize,
    maxsize,
    base_id,
    timestep,
    split_levels,
    split_seed_minsize,
    split_seed_minstrength,
    topo_filter_apply,
    topo_filter_mode,
    topo_filter_threshold,
    topo_filter_min_overlap,
    topo_fld,
    silent=False,
    grid=None,
):
    """Identify all features from a field and add some properties to them."""
    debug = False

    # Apply or prepare topo filter
    filter_mask = None
    if topo_filter_apply:
        if topo_fld is not None and topo_filter_threshold >= 0:
            topo_mask = (topo_fld > topo_filter_threshold).astype(np.uint8)
        if topo_filter_mode == "features":
            filter_mask = topo_mask
        elif topo_filter_mode == "pixels":
            fld = np.where(topo_mask > 0, 0.0, fld).astype(np.float32)

    if grid is None:
        const = default_constants(nx=nx, ny=ny)
        grid = Grid(const)
    grid.set_values(fld)

    # Identify features
    try:
        features = find_features_2d_threshold(
            fld.astype(np.float32),
            lower=float(lower) if lower is not None else -1,
            upper=float(upper) if upper is not None else -1,
            minsize=int(minsize),
            maxsize=int(maxsize),
            base_id=base_id,
            grid=grid,
        )
    except:
        raise Exception("error identifying features")
    if not silent:
        log.info(
            f"identified {len(features)} '{feature_name}' features based on range "
            f"({lower}..{upper})"
        )

    # Split features
    if split_levels is not None:
        nold = len(features)
        features = split_regiongrow_levels(
            features,
            split_levels,
            nx=nx,
            ny=ny,
            base_id=base_id,
            minsize=minsize,
            maxsize=maxsize,
            seed_minsize=split_seed_minsize,
            seed_min_strength=split_seed_minstrength,
            filter_mask=filter_mask,
            filter_min_overlap=float(topo_filter_min_overlap),
        )
        if not silent:
            log.info(
                "split {nold} '{feature_name}' features into {len(features)} "
                f"based on levels {split_levels}"
            )

    # Add properties
    if debug:
        log.debug(f"add properties to '{feature_name}' features")
    for feature in features:
        feature.timestep = timestep

    return features


# :call: > --- callers ---
# :call: > test_stormtrack::test_core::test_features::test_regions::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: v stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::constants::default_constants
def find_features_2d_threshold_seeded(
    field_raw,
    *,
    lower=None,
    upper=None,
    minsize=1,
    seeds=None,
    random_seed=None,
    nmax_cregions=10,
    nmax_cregions_pixels=10,
    nmax_cregions_shells=1,
    nmax_cregions_shell=10,
    nmax_cregions_connected=10,
    Constants constants=None,
):
    """Identify all features based on thresholds.

    Features are sets of connected pixels.

    The algorithm is designed such that it can be parallelized.

    Algorithm:
     - Select a random unassigned pixel (neither background nor feature).
     - If the pixel doesn't meet the condition, assign it to the background
       and continue with the next random unassigned pixel.
     - If the pixel meets the condition, start a new pixel region.
     - Iterate over all neighbor pixels (region growing).
     - If one meets the condition, add it to the region.
     - If one already belongs to a region, connect the regions (references).
     - Grow the region until all neighbors have been assigned, then start over.
     - Once all pixels are assigned, build features from connected regions.

    TODO:
     - Implement more general condition(s) instead of only threshold
     - Adapt docstring for non-random seeds

    """
    # Determine thresholds
    if lower is None and upper is None:
        raise ValueError("No threshold given! Must pass lower, upper, or both!")
    elif lower is None:
        lower = np.finfo(np.float32).min
    elif upper is None:
        upper = np.finfo(np.float32).max

    # Set up constants
    if constants is None:
        nx, ny = field_raw.shape
        constants = default_constants(nx=nx, ny=ny)
    else:
        nx = constants.nx
        ny = constants.ny
    cdef cConstants* cconstants = constants.to_c()

    # Call with given seeds
    if seeds is not None:
        if random_seed is not None:
            log.warning(
                "inconsistent arguments: random_seed pointless when passing seeds"
            )
        return c_find_features_2d_threshold_seeds(
            field_raw,
            lower,
            upper,
            minsize,
            seeds,
            nmax_cregions,
            nmax_cregions_pixels,
            nmax_cregions_shells,
            nmax_cregions_shell,
            nmax_cregions_connected,
            cconstants,
        )

    # Call with random seeds
    else:
        use_random_seed = (random_seed is not None)
        if random_seed is None:
            random_seed = 0
        use_random_seed = (random_seed is not None)
        return _find_features_threshold_random_seeds(
            field_raw,
            lower,
            upper,
            minsize,
            random_seed,
            use_random_seed,
            nmax_cregions,
            nmax_cregions_pixels,
            nmax_cregions_shells,
            nmax_cregions_shell,
            nmax_cregions_connected,
            cconstants,
        )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::find_features_2d_threshold_seeded
# :call: v --- calling ---
# :call: v stormtrack::core::identification::c_find_features_2d_threshold_seeds_core
# :call: v stormtrack::core::identification::cregions_create_features(
# :call: v stormtrack::core::identification::cregions_merge_connected
# :call: v stormtrack::core::identification::features_reset_cregion(
# :call: v stormtrack::core::identification::init_random_seeds
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v stormtrack::core::tables::pixel_status_table_alloc
# :call: v stormtrack::core::grid::Grid
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::grid::grid_cleanup
# :call: v stormtrack::core::grid::grid_create
cdef list _find_features_threshold_random_seeds(
    np.float32_t [:, :] field_raw,
    np.float32_t lower_threshold,
    np.float32_t upper_threshold,
    int minsize,
    int random_seed,
    bint use_random_seed,
    int nmax_cregions,
    int nmax_cregions_pixels,
    int nmax_cregions_shells,
    int nmax_cregions_shell,
    int nmax_cregions_connected,
    cConstants* constants,
):
    cdef cRegionConf cregion_conf = cRegionConf(
        connected_max=nmax_cregions_connected,
        pixels_max=nmax_cregions_pixels,
        shells_max=nmax_cregions_shells,
        shell_max=nmax_cregions_shell,
        hole_max=0,
        holes_max=0,
    )

    # Initialize grid
    cdef cGrid grid = grid_create(field_raw, constants[0])
    neighbor_link_stat_table_alloc(&grid.neighbor_link_stat_table, constants)

    # Initialize random seeds (shuffle indices)
    cdef int n_seeds = grid.constants.nx*grid.constants.ny

    # SR_TODO replace by real points
    # Initialize dummy points
    cdef np.ndarray[np.int32_t, ndim=1] dummy_center = np.array([], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] dummy_extrema = np.array([[]], dtype=np.int32)

    # Initialize seeds
    if use_random_seed:
        np.random.seed(random_seed)
    cdef np.ndarray[np.int32_t, ndim=2] random_seeds_raw = init_random_seeds(&grid)
    cdef np.int32_t** random_seeds = <np.int32_t**>malloc(n_seeds*sizeof(np.int32_t*))
    for i in range(n_seeds):
        random_seeds[i] = <np.int32_t*>malloc(3*sizeof(np.int32_t))
        random_seeds[i][0] = random_seeds_raw[i, 0]
        random_seeds[i][1] = random_seeds_raw[i, 1]
        random_seeds[i][2] = 1

    # Prepare regions
    cdef cRegions cregions = cregions_create(nmax_cregions)

    cdef int n_unassigned_pixels = grid.constants.nx*grid.constants.ny
    cdef bint selected_seeds = False

    # Identify pixels that belong to features
    c_find_features_2d_threshold_seeds_core(
        field_raw,
        &grid,
        lower_threshold,
        upper_threshold,
        minsize,
        random_seeds,
        n_seeds,
        &cregions,
        &n_unassigned_pixels,
        selected_seeds,
    )

    # Join adjacent regions into features
    cdef bint exclude_seed_points = False
    cdef cRegions cfeatures = cregions_merge_connected(
        &cregions,
        &grid,
        exclude_seed_points,
        nmax_cregions,
        cregion_conf,
        constants,
        cleanup=True,
    )

    # Allocate lookup tables
    pixel_status_table_alloc(&grid.pixel_status_table, constants)
    pixel_region_table_alloc_grid(&grid.pixel_region_table, constants)

    # Build the feature objects
    cdef np.uint64_t base_id = 0 # SR_TODO implement IDs properly
    cdef bint ignore_missing_neighbors = False
    cdef list features = cregions_create_features(
        &cfeatures,
        base_id,
        ignore_missing_neighbors,
        &grid,
        constants,
    )

    # Free allocated memory
    cregions_cleanup(&cfeatures, cleanup_regions=True)
    grid_cleanup(&grid)
    features_reset_cregion(features, warn=False)
    for i in range(n_seeds):
        free(random_seeds[i])
    free(random_seeds)

    return features


# :call: > --- callers ---
# :call: > stormtrack::core::identification::find_features_2d_threshold_seeded
# :call: v --- calling ---
# :call: v stormtrack::core::identification::c_find_features_2d_threshold_seeds_core
# :call: v stormtrack::core::identification::cregions_create_features
# :call: v stormtrack::core::identification::cregions_merge_connected
# :call: v stormtrack::core::identification::features_reset_cregion
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v stormtrack::core::tables::pixel_status_table_alloc
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::grid::grid_cleanup
# :call: v stormtrack::core::grid::grid_create
cdef list c_find_features_2d_threshold_seeds(
    np.float32_t [:, :] field_raw,
    np.float32_t lower_threshold,
    np.float32_t upper_threshold,
    int minsize,
    np.int32_t[:, :] seeds_in,
    int nmax_cregions,
    int nmax_cregions_pixels,
    int nmax_cregions_shells,
    int nmax_cregions_shell,
    int nmax_cregions_connected,
    cConstants* constants,
):
    cdef cRegionConf cregion_conf = cRegionConf(
        connected_max=nmax_cregions_connected,
        pixels_max=nmax_cregions_pixels,
        shells_max=nmax_cregions_shells,
        shell_max=nmax_cregions_shell,
        hole_max=0,
        holes_max=0,
    )
    cdef bint exclude_seed_points = False

    # Initialize grid
    cdef cGrid grid = grid_create(field_raw, constants[0])
    neighbor_link_stat_table_alloc(&grid.neighbor_link_stat_table, constants)

    # Initialize seeds indices)
    cdef int i
    cdef int n_seeds = seeds_in.shape[0]
    cdef np.int32_t** seeds = <np.int32_t**>malloc(n_seeds*sizeof(np.int32_t*))
    for i in range(n_seeds):
        seeds[i] = <np.int32_t*>malloc(3*sizeof(np.int32_t))
        seeds[i][0] = seeds_in[i, 0]
        seeds[i][1] = seeds_in[i, 1]
        seeds[i][2] = 1

    # Initialize dummy points
    cdef np.ndarray[np.int32_t, ndim=1] dummy_center = np.array([], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] dummy_extrema = np.array([[]], dtype=np.int32)

    # Prepare regions
    cdef cRegions cregions = cregions_create(nmax_cregions)

    cdef int n_unassigned_pixels = grid.constants.nx*grid.constants.ny
    cdef bint selected_seeds = True

    # Identify pixels that belong to features
    c_find_features_2d_threshold_seeds_core(
        field_raw,
        &grid,
        lower_threshold,
        upper_threshold,
        minsize,
        seeds,
        n_seeds,
        &cregions,
        &n_unassigned_pixels,
        selected_seeds,
    )

    # Join adjacent regions into features
    cdef cRegions cfeatures = cregions_merge_connected(
        &cregions,
        &grid,
        exclude_seed_points,
        nmax_cregions,
        cregion_conf,
        constants,
        cleanup=True,
    )

    # SR_TMP <
    # Allocate lookup tables
    pixel_status_table_alloc(&grid.pixel_status_table, constants)
    pixel_region_table_alloc_grid(&grid.pixel_region_table, constants)
    # SR_TMP >

    # Build the feature objects
    cdef np.uint64_t base_id = 0
    cdef bint ignore_missing_neighbors = False
    cdef list features = cregions_create_features(
        &cfeatures, base_id, ignore_missing_neighbors, &grid, constants,
    )

    # Free allocated memory
    cregions_cleanup(&cfeatures, cleanup_regions=True)
    grid_cleanup(&grid)
    features_reset_cregion(features, warn=False)
    for i in range(n_seeds):
        free(seeds[i])
    free(seeds)

    return features


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: v --- calling ---
# :call: v stormtrack::core::identification::assign_cpixel
# :call: v stormtrack::core::identification::grow_cregion_rec
# :call: v stormtrack::core::identification::pop_random_unassigned_pixel
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegions
cdef void c_find_features_2d_threshold_seeds_core(
    np.float32_t[:, :] field_raw,
    cGrid* grid,
    np.float32_t lower_threshold,
    np.float32_t upper_threshold,
    int minsize,
    np.int32_t** seeds,
    int n_seeds,
    cRegions*    cregions,
    int* n_unassigned_pixels,
    bint selected_seeds,
) except *:
    cdef bint debug = False
    if debug:
        log.debug(f"< c_find_features_2d_threshold_seeds_core ({n_seeds} seeds)")
    cdef int i
    cdef int j
    cdef int pid
    cdef int i_seeds=0
    cdef int nx = field_raw.shape[0]
    cdef int ny = field_raw.shape[1]
    cdef cPixel* cpixel

    # initialize unassigned pixels
    cdef np.int8_t* _unassigned = <np.int8_t*>malloc(nx*ny*sizeof(np.int8_t))
    cdef np.int8_t[:, :] unassigned = <np.int8_t[:nx, :ny]>_unassigned
    cdef int n_unassigned_seeds = n_seeds

    for i in range(nx):
        for j in range(ny):
            cpixel = &grid.pixels[i][j]
            unassigned[i, j] = 1
            pid = cpixel.id
    for i in range(n_seeds):
        grid.pixels[seeds[i][0]][seeds[i][1]].is_seed = True
        if seeds[i][2] == 0:
            unassigned[seeds[i][0], seeds[i][1]] = 0
            n_unassigned_seeds -= 1
    if debug:
        log.debug(f"n_unassigned_seeds0={n_unassigned_seeds}")
        log.debug(f"n_unassigned_pixels0={n_unassigned_pixels[0]}")

    # Assign all pixels to either a region or the background
    cdef int n_iter_max = 10 * grid.constants.nx * grid.constants.ny
    for i in range(n_iter_max):
        if n_unassigned_seeds <= 0: # or n_unassigned_pixels[0] <= 0:
            if debug:
                log.debug(
                    f"BREAK n_unassigned_seeds={n_unassigned_seeds}, "
                    f"n_unassigned_pixels={n_unassigned_pixels[0]}"
                )
            break

        # DBG_BLOCK <
        # if debug:
        #     log.debug("<<<")
        #     for i in range(n_seeds):
        #         log.debug(
        #             f"@ ({seeds[i][0]}, {seeds[i][1]}) "
        #             f"{seeds[i][2]} {unassigned[seeds[i][0]}"
        #         )
        #     log.debug("---")
        #     log.debug("IS_SEED ({}, {}) {}".format(cpixel.x, cpixel.y, i_seeds))
        #     log.debug("<<<")
        # DBG_BLOCK >

        # Get a random unassigned seed pixel
        if debug:
            log.debug(
                f"UNASSIGNED: {n_unassigned_seeds} seeds, "
                f"{n_unassigned_pixels[0]} pixels"
            )
        if i_seeds == n_seeds:
            if debug:
                log.debug(f"BREAK i_seeds == n_seeds ({i_seeds})")
            break
        cpixel = pop_random_unassigned_pixel(
            seeds, &i_seeds, unassigned, grid, n_seeds, selected_seeds,
        )
        n_unassigned_seeds -= 1
        if not selected_seeds:
            n_unassigned_pixels[0] -= 1

        # DBG_BLOCK <
        # if debug:
        #     log.debug(">>>")
        #     for i in range(n_seeds):
        #         log.debug(
        #             f"@ ({seeds[i][0]}, {seeds[i][1]}) "
        #             f"{seeds[i][2]} {unassigned[seeds[i][0]}"
        #         )
        #     log.debug("---")
        #     log.debug(f"IS_SEED ({cpixel.x}, {cpixel.y}) {i_seeds}")
        #     log.debug(">>>")
        # DBG_BLOCK >

        # Assign the pixel to a region or the background
        assign_cpixel(
            cpixel, lower_threshold, upper_threshold, cregions, NULL, grid,
        )
        # SR_TMP >

        # Grow the region if the pixel already belongs to one
        if cpixel.type == pixeltype_feature:
            grow_cregion_rec(
                cpixel,
                lower_threshold,
                upper_threshold,
                grid,
                unassigned,
                n_unassigned_pixels,
                &n_unassigned_seeds,
                cregions,
                selected_seeds,
            )
    else:
        err = "identify_regions: loop timed out (nmax={n_iter_max})"
        raise Exception(err)

    if debug: log.debug("> c_find_features_2d_threshold_seeds_core")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds_core
# :call: v --- calling ---
# :call: v stormtrack::core::identification::assign_cpixel
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::pixeltype
# :call: v stormtrack::core::cregion::cregion_connect_with
cdef void grow_cregion_rec(
    cPixel* cpixel_center,
    np.float32_t lower_threshold,
    np.float32_t upper_threshold,
    cGrid* grid,
    np.int8_t[:, :] unassigned,
    int* n_unassigned_pixels,
    int* n_unassigned_seeds,
    cRegions* cregions,
    bint selected_seeds,
) except *:
    cdef bint debug = False
    if debug:
        log.debug("< grow_cregion_rec")
    cdef cPixel* other_cpixel
    cdef int i
    cdef int neighbor_pid
    cdef pixeltype ctype

    # DBG_BLOCK <
    if debug:
        log.debug(f"CREGIONS: {cregions.n}")
        for i in range(cregions.n):
            log.debug(
                f" => [{cregions.regions[i].id}] "
                f"{cregions.regions[i].pixels_n}/{cregions.regions[i].pixels_max}, "
                f"{cregions.regions[i].connected_n}/{cregions.regions[i].connected_max}"
            )
    # DBG_BLOCK >

    if debug:
        log.debug(f"N_NEIGHBORS {cpixel_center.neighbors_n}")
    for i in range(cpixel_center.neighbors_max):
        other_cpixel = cpixel_center.neighbors[i]
        if other_cpixel is NULL:
            continue
        ctype = other_cpixel.type
        if debug:
            log.debug(
                f"OTHER PIXEL: ({other_cpixel.x}, {other_cpixel.y}) "
                f"[{i}/{cpixel_center.neighbors_max}]"
            )

        if ctype == pixeltype_background:
            if debug:
                log.debug(
                    f" -> other pixel is background: "
                    f"({other_cpixel.x}, {other_cpixel.y})"
                )
            continue

        elif ctype == pixeltype_feature:
            if debug:
                log.debug(
                    f" -> other pixel is feature: ({other_cpixel.x}, {other_cpixel.y})"
                )
            if other_cpixel.region != cpixel_center.region:
                if debug:
                    log.debug(
                        f"< connect {cpixel_center.region.id} {other_cpixel.region.id}"
                    )
                cregion_connect_with(cpixel_center.region, other_cpixel.region)

        elif ctype == pixeltype_none:
            if debug:
                log.debug(
                    f" -> other pixel is unassigned: "
                    f"({other_cpixel.x}, {other_cpixel.y})"
                )
            unassigned[other_cpixel.x, other_cpixel.y] = -1
            assign_cpixel(
                other_cpixel,
                lower_threshold,
                upper_threshold,
                cregions,
                cpixel_center.region,
                grid,
            )
            ctype = other_cpixel.type
            if selected_seeds:
                if ctype == pixeltype_feature:
                    n_unassigned_pixels[0] -= 1
            else:
                n_unassigned_seeds[0] -= 1
                n_unassigned_pixels[0] -= 1

            if debug:
                log.debug(
                    f"assigned other pixel to "
                    f"{'feature' if ctype == pixeltype_feature else 'background'}"
                )
            # SR_TODO think about this recursive calls (it is not necessary
            # SR_TODO as the result is the same if it is just removed/commented;
            # SR_TODO and there might be a better, more intuitive approach
            # SR_TODO (recursion sucks) to the region growing (i.e. a loop)

            if ctype == pixeltype_feature:
                if debug:
                    log.debug(
                        f" -> other pixel is feature: "
                        f"({other_cpixel.x}, {other_cpixel.y})"
                    )
                grow_cregion_rec(
                    other_cpixel,
                    lower_threshold,
                    upper_threshold,
                    grid,
                    unassigned,
                    n_unassigned_pixels,
                    n_unassigned_seeds,
                    cregions,
                    selected_seeds,
                )
    if debug:
        log.debug("> grow_cregion_rec")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
cdef np.ndarray[np.int32_t, ndim=2] init_random_seeds(cGrid* grid):
    cdef np.ndarray[np.int32_t, ndim=2] inds = np.empty(
            [grid.constants.nx*grid.constants.ny, 2], dtype=np.int32,
    )
    cdef int i
    cdef int j
    for i in range(grid.constants.nx):
        for j in range(grid.constants.ny):
            inds[i*grid.constants.ny + j, 0] = i
            inds[i*grid.constants.ny + j, 1] = j
    inds = np.random.permutation(inds)
    return inds


# :call: > --- callers ---
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
@cython.boundscheck(False)
@cython.wraparound(False)
cdef cPixel* pop_random_unassigned_pixel(
    np.int32_t** seeds,
    int* i_seeds,
    np.int8_t[:, :] unassigned,
    cGrid* grid,
    int n_seeds,
    bint selected_seeds,
) except NULL:
    cdef int i
    cdef int x
    cdef int y
    cdef int z
    for i in range(i_seeds[0], n_seeds):
        x = seeds[i][0]
        y = seeds[i][1]
        z = seeds[i][2]
        if z <= 0:
            continue
        seeds[i][2] = 0
        if not selected_seeds and unassigned[x, y] < 0:
            continue
        unassigned[x, y] = -1
        i_seeds[0] = i + 1
        return &grid.pixels[x][y]
    err = "no random unassigned pixel found"
    raise Exception(err)


# :call: > --- callers ---
# :call: v --- calling ---
@cython.cdivision(True)
cdef inline int random_int(int min, int max):
    # srand(42) # for testing
    return int(rand() / RAND_MAX * (max - 1))


# :call: > --- callers ---
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::cregions_merge_connected
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregions::cregions_move
cdef void cregions_merge_connected_inplace(
    cRegions* cregions,
    cGrid* grid,
    bint exclude_seed_points,
    int nmax_cregions,
    cRegionConf cfeature_conf,
    cConstants* constants,
) except *:
    if cregions.n <= 1:
        return
    cdef cRegions cregions_merged = cregions_merge_connected(
        cregions,
        grid,
        exclude_seed_points,
        nmax_cregions,
        cfeature_conf,
        constants,
        cleanup=False,
    )
    cregions_move(&cregions_merged, cregions)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::cregions_merge_connected_inplace
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: v --- calling ---
# :call: v stormtrack::core::identification::_cregions_merge_connected_core
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
cdef cRegions cregions_merge_connected(
    cRegions* cregions,
    cGrid* grid,
    bint exclude_seed_points,
    int nmax_cregions,
    cRegionConf cfeature_conf,
    cConstants* constants,
    bint cleanup,
) except *:
    cdef bint debug = False
    if debug:
        log.debug("< cregions_merge_connected")
    cdef int pid
    cdef int i
    cdef int j
    cdef cPixel** pixels_feature = <cPixel**>malloc(
        constants.nx*constants.ny*sizeof(cPixel*)
    )

    # Initialize output cregions
    cdef cRegions cregions_out = cregions_create(nmax_cregions)

    cdef list features = []
    cdef int n_connected_regions

    # Initialized lookup table for processed pixels
    cdef bint** pixel_added = <bint**>malloc(constants.nx*sizeof(bint*))
    for i in range(constants.nx):
        pixel_added[i] = <bint*>malloc(constants.ny*sizeof(bint))
        for j in range(constants.ny):
            pixel_added[i][j] = False

    # Initialize temporary array for unprocessed cregions
    cdef cRegion** cregions_todo = <cRegion**>malloc(cregions.n*sizeof(cRegion*))
    for i in range(cregions.n):
        cregions_todo[i] = cregions.regions[i]

    # Prepare connected regions
    cdef int connected_regions_max = cregions.n
    cdef cRegion** connected_regions = <cRegion**>malloc(
        connected_regions_max*sizeof(cRegion*)
    )
    for i in range(connected_regions_max):
        connected_regions[i] = NULL

    # Call core function
    _cregions_merge_connected_core(
        &cregions_out,
        cregions,
        cregions_todo,
        connected_regions,
        connected_regions_max,
        pixels_feature,
        pixel_added,
        grid,
        exclude_seed_points,
        constants,
    )

    # Cleanup
    free(cregions_todo)
    free(connected_regions)
    for i in range(constants.nx):
        free(pixel_added[i])
    free(pixel_added)
    free(pixels_feature)
    if cleanup:
        cregions_cleanup(cregions, cleanup_regions=True)

    if debug:
        log.debug("> cregions_merge_connected")
    return cregions_out


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregions_merge_connected
# :call: v --- calling ---
# :call: v stormtrack::core::identification::collect_pixels
# :call: v stormtrack::core::identification::cregion_collect_connected_regions
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::grid::grid_create_cregion
cdef void _cregions_merge_connected_core(
    cRegions* cregions_out,
    cRegions* cregions,
    cRegion** cregions_todo,
    cRegion** connected_regions,
    int connected_regions_max,
    cPixel** pixels_feature,
    bint** pixel_added_to_feature,
    cGrid* grid,
    bint exclude_seed_points,
    cConstants* constants,
) except *:
    cdef str _name_ = "_cregions_merge_connected_core"
    cdef bint debug = False
    cdef int i
    cdef int j
    cdef cRegion* cregion_start
    cdef int cregions_n = cregions.n
    cdef int connected_regions_n = 0
    cdef int n_pixels_feature
    cdef cRegion* cfeature

    # Loop until no more ids are left
    cdef int i_next_region = 0
    cdef int iter_i
    cdef int iter_max=1000000
    for iter_i in range(iter_max):
        if debug:
            log.debug(f"\n+++ ITER {iter_i} +++")
        for i in range(cregions_n):
            if cregions_todo[i] is not NULL:
                if debug:
                    log.debug(f"cregions.regions[{i}] is not NULL")
                break
        else:
            if debug:
                log.debug("BREAK all {cregions_n} regions processed")
            break

        # Select first unprocessed region (start region for next feature)
        for i in range(cregions_n):
            if cregions_todo[i] is not NULL:
                cregion_start = cregions.regions[i]
                break

        # Collect all regions connected to the start region
        for i in range(connected_regions_n):
            connected_regions[i] = NULL
        connected_regions_n = cregion_collect_connected_regions(
            cregions, connected_regions, connected_regions_max, cregion_start,
        )
        if debug:
            log.debug(f"connected_regions.n={connected_regions_n}")

        # Remove newly connected regions from todo list (so to speak)
        for i in range(connected_regions_n):
            for j in range(cregions_n):
                if cregions_todo[j] is NULL:
                    continue
                if cregions_todo[j].id == connected_regions[i].id:
                    cregions_todo[j] = NULL
                    break

        # Collect all pixels of these connected regions
        n_pixels_feature = collect_pixels(
            connected_regions,
            connected_regions_n,
            pixels_feature,
            cregions,
            grid,
            pixel_added_to_feature,
            exclude_seed_points,
        )
        if debug:
            log.debug(f"n_pixels_feature={n_pixels_feature}")

        # Check whether there are no pixels (which might not be the case
        # for exclude_seed_points=True if only seed points have been found)
        if n_pixels_feature == 0:
            continue

        # Create new feature from the collected pixels
        cfeature = grid_create_cregion(grid)
        cregions_link_region(cregions_out, cfeature, cleanup=False, unlink_pixels=False)
        for i in range(n_pixels_feature):
            cregion_insert_pixel(
                cfeature, pixels_feature[i], link_region=True, unlink_pixel=True,
            )
        cregion_determine_boundaries(cfeature, grid)
    else:
        raise Exception(f"loop timed out after {iter_i} iterations")
    if debug:
        log.debug(f"\n+++ LOOP DONE +++ in {iter_i} iterations")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
cdef int collect_pixels(
    cRegion** connected_regions,
    int connected_regions_n,
    cPixel** pixels_feature,
    cRegions* cregions,
    cGrid* grid,
    bint** pixel_added_to_feature,
    bint exclude_seed_points,
) except -999:
    """Collect all pixels of multiple regions in a pre-alloc'd array."""
    cdef int i
    cdef int j
    cdef int k
    cdef int i_pixels_feature=0
    cdef cRegion* cregion
    cdef cPixel* cpixel
    # Process regions one by one
    for i in range(connected_regions_n):
        cregion = connected_regions[i]
        # Collect pixels of region
        for j in range(cregion.pixels_max):
            cpixel = cregion.pixels[j]
            if cpixel is NULL:
                continue
            if not pixel_added_to_feature[cpixel.x][cpixel.y]:
                pixel_added_to_feature[cpixel.x][cpixel.y] = True
                if cpixel.is_seed and exclude_seed_points:
                    continue
                pixels_feature[i_pixels_feature] = cpixel
                i_pixels_feature += 1
    return i_pixels_feature


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::structs::cPixel
# SR_TODO eliminate (only used in old cyclone id code)
cdef Feature create_feature(
    cPixel** pixels_feature,
    int n_pixels_feature,
    np.int32_t[:] center,
    np.int32_t[:, :] extrema,
    np.uint64_t feature_id,
):
    log.warning("TODO remove function create_feature")
    # print(f"< create_feature {feature_id}")
    cdef np.ndarray[np.int32_t, ndim=2] pixels = np.empty(
        [n_pixels_feature, 2], dtype=np.int32,
    )
    cdef np.ndarray[np.float32_t, ndim=1] values = np.empty(
        [n_pixels_feature], dtype=np.float32,
    )
    cdef int i
    cdef int j
    cdef cPixel* cpixel
    cdef Feature feature
    for i in range(n_pixels_feature):
        cpixel = pixels_feature[i]
        pixels[i, 0] = cpixel.x
        pixels[i, 1] = cpixel.y
        values[i] = cpixel.v
    feature = Feature(
        values=values,
        pixels=pixels,
        center=center,
        extrema=extrema,
        id_=feature_id,
    )
    # print(f"> create_feature {feature_id}")
    return feature


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: v --- calling ---
# :call: v stormtrack::core::identification::_cregion_collect_connected_regions_rec
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
cdef int cregion_collect_connected_regions(
    cRegions* cregions,
    cRegion** connected_regions,
    int connected_regions_max,
    cRegion* cregion_start,
) except -999:
    # SR_TODO better solution than bare-bones pointer array
    if connected_regions_max < cregions.n:
        raise Exception(
            f"array connected_regions potentially too small "
            f"({connected_regions_max} < {cregions.n})"
        )
    cdef int connected_regions_n = 0
    _cregion_collect_connected_regions_rec(
        cregions,
        connected_regions,
        &connected_regions_n,
        connected_regions_max,
        cregion_start,
    )
    return connected_regions_n


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregion_collect_connected_regions
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
cdef void _cregion_collect_connected_regions_rec(
    cRegions* cregions,
    cRegion** connected_regions,
    int* connected_regions_n,
    int connected_regions_max,
    cRegion* cregion,
) except *:
    cdef int i

    # Check if region has already been collected
    for i in range(connected_regions_n[0]):
        if connected_regions[i].id == cregion.id:
            return

    # Add region to array
    if connected_regions_n[0] == connected_regions_max:
        raise Exception(
            f"_cregion_collect_connected_regions_rec: connected_regions not big enough "
            f"(max={connected_regions_max})"
        )
    connected_regions[connected_regions_n[0]] = cregion
    connected_regions_n[0] += 1

    # Continue with all connected regions
    cdef cRegion* connected_region
    for i in range(cregion.connected_n):
        connected_region = cregion.connected[i]

        # Check if this point has already been passed
        if connected_region is NULL:
            continue

        # SR_DBG looks like this is necessary for the above NULL check
        # SR_DBG to work, but I'm not really sure whether this is correct...
        cregion.connected[i] = NULL # SR_DBG

        # Continue with previously unprocessed region
        _cregion_collect_connected_regions_rec(
            cregions,
            connected_regions,
            connected_regions_n,
            connected_regions_max,
            connected_region,
        )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds_core
# :call: > stormtrack::core::identification::grow_cregion_rec
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cpixel_set_region
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::grid::grid_create_cregion
cdef void assign_cpixel(
    cPixel* cpixel,
    np.float32_t lower_threshold,
    np.float32_t upper_threshold,
    cRegions* cregions,
    cRegion* cregion,
    cGrid* grid,
) except *:
    cdef bint debug = False
    # if debug:
    #     log.debug(f"< assign_cpixel ({cpixel.x}, {cpixel.y})")

    # If the pixel does not fulfil the feature condition,
    # assign it to the background and we're already done
    if cpixel.v < lower_threshold or cpixel.v >= upper_threshold:
        cpixel.type = pixeltype_background
        if debug:
            log.debug(f"> assign_cpixel ({cpixel.x}, {cpixel.y}) BACKGROUND")
        return

    # If the pixel does fulfil the feature condition, assign it to the
    # current region if there is one, or to a new region otherwise
    cpixel.type = pixeltype_feature
    if cregion is NULL:
        cregion = grid_create_cregion(grid)
        cregions_link_region(cregions, cregion, cleanup=False, unlink_pixels=False)
    cpixel_set_region(cpixel, cregion)
    cdef bint link_region=False, unlink_pixel=False
    cregion_insert_pixel(cregion, cpixel, link_region, unlink_pixel)
    if debug:
        log.debug(f"> assign_cpixel ({cpixel.x}, {cpixel.y}) FEATURE {cregion.id}")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::identify_features
# :call: > test_stormtrack::test_core::test_features::test_regions::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::cregions_create_features
# :call: v stormtrack::core::identification::eliminate_regions_by_size
# :call: v stormtrack::core::identification::features_reset_cregion
# :call: v stormtrack::core::identification::find_existing_region
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: v stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v stormtrack::core::tables::pixel_region_table_reset
# :call: v stormtrack::core::tables::pixel_status_table_alloc
# :call: v stormtrack::core::tables::pixel_status_table_reset
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::grid::Grid
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::constants::default_constants
# :call: v stormtrack::core::grid::grid_create_cregion
def find_features_2d_threshold(
    np.float32_t[:, :] field_raw,
    *,
    float lower=-1,
    float upper=-1,
    int minsize=1,
    int maxsize=-1,
    int trim_bnd_n=0,
    np.uint64_t base_id=0,
    int nmax_cregion=10,
    int nmax_cregion_pixels=50,
    int nmax_cregion_shells=1,
    int nmax_cregion_shell=20,
    int nmax_cregion_connected=5,
    Grid grid=None,
    Constants constants=None,
):
    """Find all regions of pixels which fulfil a certain condition."""
    cdef bint debug = False
    if debug:
        log.debug("< find_features_threshold")

    # Check thresholds (at least one must be passed)
    if lower >= 0 and upper >= 0:
        mode = 0
    elif lower < 0 and upper >=0:
        mode = -1
        lower = 0
    elif lower >= 0 and upper < 0:
        mode = 1
        upper = 0
    else: # lower < 0 and upper < 0
        raise ValueError("no threshold given (neither lower nor upper)")

    # Set up constants
    cdef int i
    cdef int j
    cdef int k
    cdef np.int32_t nx = field_raw.shape[0]
    cdef np.int32_t ny = field_raw.shape[1]
    cdef cPixel* cpixel
    cdef cRegion* cregion
    cdef cRegionConf cregion_conf = cRegionConf(
        connected_max=nmax_cregion_connected,
        pixels_max=nmax_cregion_pixels,
        shells_max=nmax_cregion_shells,
        shell_max=nmax_cregion_shell,
        hole_max=0,
        holes_max=0,
    )
    cdef cRegions cregions = cregions_create(nmax_cregion)
    cdef int next_rid = 0

    # Set up constants and grid
    if constants is None:
        constants = default_constants(nx=nx, ny=ny)
    if grid is None:
        grid = Grid(constants)
        grid.set_values(np.asarray(field_raw))
    cdef cGrid* cgrid = grid.to_c()
    cdef cConstants* cconstants = grid.constants.to_c()

    # Identify regions
    if debug:
        log.debug(
            f"identify regions ("
            f"{'' if mode == -1 else lower}..{'' if mode == 1 else lower})"
        )
    cdef int loop_order = 0 # 0: ij; 1: ji SR_TMP
    for i in range(cconstants.nx):
        for j in range(cconstants.ny):
            with cython.boundscheck(False), cython.wraparound(False):
                if mode == 1 and not lower <= field_raw[i, j]:
                    continue
                if mode == -1 and not field_raw[i, j] < upper:
                    continue
                if mode == 0 and not lower <= field_raw[i, j] < upper:
                    continue
                cpixel = &cgrid.pixels[i][j]
            cregion = find_existing_region(cgrid, i, j, loop_order)
            if cregion is NULL:
                cregion = grid_create_cregion(cgrid)
                cregions_link_region(
                    &cregions, cregion, cleanup=False, unlink_pixels=False,
                )
                if debug:
                    log.debug(f"new region {cregion.id}")
            if debug:
                log.debug(f"[{cregion.id}] ({cpixel.x}, {cpixel.y}) {cpixel.v}")
            cregion_insert_pixel(cregion, cpixel, link_region=True, unlink_pixel=False)

    # Eliminate too small regions
    cdef int nold = cregions.n
    eliminate_regions_by_size(&cregions, minsize, maxsize)
    if debug:
        log.debug("eliminated too small regions: {nold} -> {cregions.n}")

    # SR_TMP < TODO integrate into Grid class
    # Allocate or reset lookup tables
    if cgrid.neighbor_link_stat_table is NULL:
        neighbor_link_stat_table_alloc(
            &cgrid.neighbor_link_stat_table, &cgrid.constants,
        )
    else:
        neighbor_link_stat_table_reset(cgrid.neighbor_link_stat_table, &cgrid.constants)
    if cgrid.pixel_status_table is NULL:
        pixel_status_table_alloc(&cgrid.pixel_status_table, &cgrid.constants)
    else:
        pixel_status_table_reset(
            cgrid.pixel_status_table, cgrid.constants.nx, cgrid.constants.ny,
        )
    if cgrid.pixel_region_table is NULL:
        pixel_region_table_alloc_grid(&cgrid.pixel_region_table, &cgrid.constants)
    else:
        pixel_region_table_reset(
            cgrid.pixel_region_table, cgrid.constants.nx, cgrid.constants.ny,
        )
    # SR_TMP >

    # Determine boundaries (both shells and holes)
    if debug:
        log.debug("determine boundaries")
    cregions_determine_boundaries(&cregions, cgrid)

    # Create Feature objects
    if debug:
        log.debug("create feature objects")
    cdef bint ignore_missing_neighbors = False
    cdef list features = cregions_create_features(
        &cregions,
        base_id,
        ignore_missing_neighbors,
        cgrid,
        cconstants,
    )

    features_reset_cregion(features, warn=False)

    if debug: log.debug("> find_features_threshold")
    return features


# :call: > --- callers ---
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_cleanup
cdef void eliminate_regions_by_size(
    cRegions* cregions, int minsize, int maxsize,
) except *:
    # print(f"< eliminate_regions_by_size (n < {minsize} | n > {maxsize})")
    cdef int i
    cdef int nold = cregions.n
    cdef int nnew = 0
    cdef cRegion** tmp = <cRegion**>malloc(nold*sizeof(cRegion*))
    for i in range(nold):
        if cregions.regions[i] is NULL:
            continue
        if (
            cregions.regions[i].pixels_n < max(1, minsize)
            or (maxsize > 0 and cregions.regions[i].pixels_n > maxsize)
        ):
            # SR_TODO is reset_connected necessary?
            cregion_cleanup(
                cregions.regions[i], unlink_pixels=True, reset_connected=True,
            )
        else:
            tmp[nnew] = cregions.regions[i]
            nnew += 1
    free(cregions.regions)
    cregions.regions = <cRegion**>malloc((nnew + 1)*sizeof(cRegion*))
    cregions.max = nnew + 1
    cregions.n = nnew
    for i in range(nnew):
        cregions.regions[i] = tmp[i]
    free(tmp)
    # print(f"> eliminate_regions_by_size ({nold} -> {nnew})")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_merge
# SR_TODO: Consider 4 vs. 8 connectivity
cdef inline cRegion* find_existing_region(
    cGrid* grid, np.int32_t x, np.int32_t y, int loop_order,
) except? NULL:
    """Find existing regions among neighbors considering loop over nx/ny.

    For 4-connectivity, only the W and S neighbors are candidates

    For 8-connectivity, in addition to the SW neighbor, either the NW or SE
    neighbor is a candidate as well, depending on the loop order which is an
    input parameter.

    """
    # SR_TMP <
    cdef int loop_order_ij = 0
    cdef int loop_order_ji = 1
    # SR_TMP >
    cdef bint debug = False
    if debug:
        log.debug(
            f"< find_existing_region ({x}, {y}) ({grid.constants.connectivity}c)"
        )

    # Check validity of loop order parameter
    if loop_order != loop_order_ij and loop_order != loop_order_ji:
        raise Exception(f"invalid loop order '{loop_order}' (must be 0=ij or 1=ji)")

    cdef cRegion* cregion    = NULL
    cdef cRegion* cregion_i  = NULL
    cdef cRegion* cregion_w  = NULL
    cdef cRegion* cregion_s  = NULL
    cdef cRegion* cregion_sw = NULL
    cdef cRegion* cregion_se = NULL # loop order ij
    cdef cRegion* cregion_nw = NULL # loop order ji

    # Get regions of preceding neighbors
    if grid.constants.connectivity >= 4:
        if x > 0:
            cregion_w = grid.pixels[x-1][y].region
        if y > 0:
            cregion_s = grid.pixels[x][y-1].region
    if grid.constants.connectivity >= 8:
        if x > 0 and y > 0:
            cregion_sw = grid.pixels[x-1][y-1].region
        if loop_order == loop_order_ij and x > 0 and y < grid.constants.ny-1:
            cregion_nw = grid.pixels[x-1][y+1].region
        if loop_order == loop_order_ji and x < grid.constants.nx-1 and y > 0:
            cregion_se = grid.pixels[x+1][y-1].region

    # DBG_BLOCK <
    if debug:
        log.debug(
            f"3 cregion_se ({x + 1}, {y - 1}) "
            f"{'NULL' if cregion_se is NULL else cregion_se.id}"
        )
        log.debug(
            f"4 cregion_s  ({x + 0}, {y - 1}) "
            f"{'NULL' if cregion_s  is NULL else cregion_s.id}"
        )
        log.debug(
            f"5 cregion_sw ({x - 1}, {y - 1}) "
            f"{'NULL' if cregion_sw is NULL else cregion_sw.id}"
        )
        log.debug(
            f"6 cregion_w  ({x - 1}, {y + 0}) "
            f"{'NULL' if cregion_w  is NULL else cregion_w.id}"
        )
        log.debug(
            f"7 cregion_nw ({x - 1}, {y + 1}) "
            f"{'NULL' if cregion_nw is NULL else cregion_nw.id}"
        )
    # DBG_BLOCK >

    # Find suitable existing region if there is one
    cdef int i
    for i in range(3, 8):
        if i == 3:
            cregion_i = cregion_se
        elif i == 4:
            cregion_i = cregion_s
        elif i == 5:
            cregion_i = cregion_sw
        elif i == 6:
            cregion_i = cregion_w
        elif i == 7:
            cregion_i = cregion_nw
        if debug:
            log.debug(
                f"{i} cregion: {'NULL' if cregion_i is NULL else cregion.id}, "
                f"cregion_i: {'NULL' if cregion_i is NULL else cregion_i.id}"
            )
        if cregion_i is not NULL:
            if cregion is NULL:
                cregion = cregion_i
            elif cregion.id != cregion_i.id:
                if debug:
                    log.debug(f" -> merge cregions {cregion.id} and {cregion_i.id}")
                cregion = cregion_merge(cregion, cregion_i)

    # DBG_BLOCK <
    if debug:
        log.debug(f"> find_existing_region {'NULL' if cregion is NULL else cregion.id}")
    # DBG_BLOCK >

    return cregion


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::TrackFeatureMerger::run
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::identification::cregions_create_features
# :call: v stormtrack::core::identification::cregions_merge_connected
# :call: v stormtrack::core::identification::features_reset_cregion
# :call: v stormtrack::core::identification::features_to_cregions
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregions::cregions_find_connected
# :call: v stormtrack::core::constants::default_constants
# :call: v stormtrack::core::grid::grid_cleanup
# :call: v stormtrack::core::grid::grid_create
# SR_TODO add option to only merge features of the connecting pixels exceed
# SR_TODO a certaion threshold (requires passing the field/values, obviously)
def merge_adjacent_features(
    list features,
    np.uint64_t base_id,
    *,
    bint find_neighbors,
    bint ignore_missing_neighbors,
    np.int32_t nx=0,
    np.int32_t ny=0,
    Constants constants=None,
):
    # print("< merge_adjacent_features")
    cdef int i
    cdef int j

    if len(features) == 0:
        return []

    # SR_TMP <
    cdef Feature feature
    cdef np.uint64_t timestep = features[0].timestep
    for feature in features:
        if feature.timestep != timestep:
            raise Exception(
                f"merge_adjacent_features: inconsistent timesteps: "
                f"{', '.join([str(f.timestep) for f in features]))}"
            )
    # SR_TMP >

    # Set up constants
    if constants is None:
        if nx == 0 or ny == 0:
            raise Exception("must pass either (nx, ny) or constants")
        constants = default_constants(nx=nx, ny=ny)
    else:
        nx = constants.nx
        ny = constants.ny
    cdef cConstants* cconstants = constants.to_c()

    # Initialize grid
    cdef np.ndarray[np.float32_t, ndim=2] field = np.zeros(
        [cconstants.nx, cconstants.ny], dtype=np.float32,
    )
    cdef cGrid grid = grid_create(field, cconstants[0])

    # Allocate lookup tables
    neighbor_link_stat_table_alloc(&grid.neighbor_link_stat_table, cconstants)
    pixel_region_table_alloc_grid(&grid.pixel_region_table, cconstants)

    # Turn features into cregions
    cdef int nmax_cregions = 10
    cdef int n_features = len(features)
    cdef cRegions cfeatures = cregions_create(nmax_cregions)
    features_to_cregions(
        features,
        n_features,
        &cfeatures,
        cregion_conf_default(),
        ignore_missing_neighbors,
        &grid,
        cconstants,
    )

    # If required, re-determine the neighbors
    cdef bint reset_existing = False
    if find_neighbors:
        cregions_find_connected(&cfeatures, reset_existing, cconstants)

    # Merge connected regions
    cdef bint exclude_seed_points = False
    cdef cRegions merged_cfeatures = cregions_merge_connected(
        &cfeatures,
        &grid,
        exclude_seed_points,
        nmax_cregions,
        cregion_conf_default(),
        cconstants,
        cleanup=True,
    )

    # Turn cregion into Feature objects
    cdef list merged_features = cregions_create_features(
        &merged_cfeatures, base_id, ignore_missing_neighbors, &grid, cconstants,
    )

    # SR_TMP <
    for feature in merged_features:
        feature.cleanup_cregion()
    # SR_TMP >

    # SR_TMP <
    for feature in merged_features:
        feature.timestep = timestep
    # SR_TMP >

    # Cleanup
    grid_cleanup(&grid)
    features_reset_cregion(features, warn=False)
    features_reset_cregion(merged_features, warn=False)

    # print("> merge_adjacent_features")
    return merged_features


# :call: > --- callers ---
# :call: > test_stormtrack::test_core::test_features::test_split_regiongrow::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::identification::_replace_feature_associations
# :call: v stormtrack::core::identification::cregions_create_features
# :call: v stormtrack::core::identification::csplit_regiongrow_core
# :call: v stormtrack::core::identification::feature_to_cregion
# :call: v stormtrack::core::identification::features_reset_cregion
# :call: v stormtrack::core::identification::features_to_cregions
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v stormtrack::core::tables::pixel_status_table_alloc
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::cregion::cregion_remove_pixel
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::constants::default_constants
# :call: v stormtrack::core::grid::grid_cleanup
# :call: v stormtrack::core::grid::grid_create
cpdef list feature_split_regiongrow(
    Feature feature,
    list seed_features,
    np.int32_t nx=0,
    np.int32_t ny=0,
    np.uint64_t base_id=0,
    list used_ids=None,
    Constants constants=None,
):
    cdef bint debug = False
    if debug:
        log.debug("< feature_split_regiongrow")

    # Check uniqueness of feature ids
    fids = set([feature.id] + [f.id for f in seed_features])
    if len(fids) <= len(seed_features):
        fids = ", ".join(
            [str(fid) for fid in [feature.id] + [f.id for f in seed_features]]
        )
        raise ValueError(f"duplicate feature ids: {fids}")

    # Set up constants
    if constants is None:
        if nx == 0 or ny == 0:
            raise Exception("must pass either (nx, ny) or constants")
        constants = default_constants(nx=nx, ny=ny)
    else:
        nx = constants.nx
        ny = constants.ny
    cdef cConstants* cconstants = constants.to_c()

    # Initialize grid
    if debug: log.debug("  initialize grid")
    cdef np.ndarray[np.float32_t, ndim=2] field = np.zeros([nx, ny], dtype=np.float32)
    cdef cGrid grid = grid_create(field, cconstants[0])

    # Allocate lookup tables
    if debug: log.debug("  allocate lookup tables")
    neighbor_link_stat_table_alloc(&grid.neighbor_link_stat_table, cconstants)
    pixel_region_table_alloc_grid(&grid.pixel_region_table, cconstants)
    pixel_status_table_alloc(&grid.pixel_status_table, cconstants)

    # Turn feature into cregion
    if debug: log.debug("  turn feature into cregion")
    cdef cRegion cfeature
    cdef bint determine_boundaries = True
    feature_to_cregion(
        feature,
        &cfeature,
        cregion_conf_default(),
        &grid,
        determine_boundaries,
        cconstants,
    )

    # Prepare lookup table for feature pixels used below to
    # remove seed pixels that don't overlap with the feature
    # The table must be initialized before the seed features
    # are converted to cregions because in the process the
    # overlapping pixels are removed from cfeature...
    # (Certainly not ideal, but whatever, that's how it is!)
    if debug:
        log.debug(f"  prepare lookup table for {feature.n} feature pixels")
    cdef int i
    cdef int j
    cdef cPixel* cpixel
    cdef bint** feature_pixel_table = <bint**>malloc(nx*sizeof(bint*))
    for i in range(nx):
        feature_pixel_table[i] = <bint*>malloc(ny*sizeof(bint))
        for j in range(ny):
            feature_pixel_table[i][j] = False
    for i in range(cfeature.pixels_istart, cfeature.pixels_iend):
        cpixel = cfeature.pixels[i]
        if cpixel is not NULL:
            feature_pixel_table[cpixel.x][cpixel.y] = True

    # Turn seeds into cregions
    cdef int n_seed_regions = len(seed_features)
    if debug: log.debug("  turn {} seeds into cregions".format(n_seed_regions))
    cdef cRegions cregions_seeds = cregions_create(n_seed_regions)
    cdef bint ignore_missing_neighbors = True
    features_to_cregions(
        seed_features,
        n_seed_regions,
        &cregions_seeds,
        cregion_conf_default(),
        ignore_missing_neighbors,
        &grid,
        cconstants,
    )

    # Remove seed points that don't overlap with feature to be split
    # Otherwise, the split features will also grow into the seed regions
    # outside of the original feature which is split
    if debug:
        log.debug("  remove non-overlapping seed points")
    cdef cRegion* cregion
    for i in range(cregions_seeds.n):
        cregion = cregions_seeds.regions[i]
        for j in range(cregion.pixels_istart, cregion.pixels_iend):
            cpixel = cregion.pixels[j]
            if cpixel is not NULL:
                if not feature_pixel_table[cpixel.x][cpixel.y]:
                    cregion_remove_pixel(cregion, cpixel)
    for i in range(nx):
        free(feature_pixel_table[i])
    free(feature_pixel_table)

    # Create cregions for new subregions
    if debug:
        log.debug("  create cregions for new subregions")
    cdef cRegions subfeature_cregions = cregions_create(n_seed_regions)

    # Grow regions
    if debug: log.debug("  grow regions")
    csplit_regiongrow_core(
        &cfeature, &cregions_seeds, &subfeature_cregions, n_seed_regions, &grid,
    )

    # Turn cregion into Feature objects
    if debug: log.debug("  turn cregion into Feature objects")
    ignore_missing_neighbors = False
    cdef list subfeatures = cregions_create_features(
        &subfeature_cregions,
        base_id,
        ignore_missing_neighbors,
        &grid,
        cconstants,
        used_ids=used_ids
    )

    # Consistency check: number of pixels must be conserved
    if sum([f.n for f in subfeatures]) != feature.n:
        raise Exception(
            f"splitting of {feature.id} failed: total number of pixels wrong: "
            f"{'+'.join([str(f.n) for f in subfeatures])} != {feature.n}"
        )

    # Clean up memory
    if debug:
        log.debug(" clean up memory")
    cregions_cleanup(&cregions_seeds, cleanup_regions=True)
    grid_cleanup(&grid)
    feature.reset_cregion(warn=False)
    features_reset_cregion(seed_features, warn=False)
    features_reset_cregion(subfeatures, warn=False)

    # SR_TODO Consider adding feature type names as optional function arguments
    # SR_TODO Actually, I should add the type name as a Feature property!!

    if debug:
        log.debug("  replace feature associations")
    _replace_feature_associations(feature, seed_features, subfeatures)

    if debug:
        log.debug("> csplit_regiongrow")
    return subfeatures


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::FeatureTracker::extend_tracks
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::identification::cfeatures_grow_core
# :call: v stormtrack::core::identification::cpixel2arr
# :call: v stormtrack::core::identification::cregions_create_features
# :call: v stormtrack::core::identification::features_reset_cregion
# :call: v stormtrack::core::identification::features_to_cregions
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::grid::Grid
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
cpdef list features_grow(
    int n,
    list features,
    Constants constants,
    bint inplace=False,
    bint retain_orig=True,
    np.uint64_t base_id=0,
    list used_ids=None,
    Grid grid=None,
):

    cdef Feature feature
    cdef cRegion* cregion
    cdef int i
    cdef int j

    # Check uniqueness of feature ids
    fids = set([f.id for f in features])
    if len(fids) < len(features):
        raise ValueError(
            f"duplicate feature ids: "
            f"{', '.join([str(fid) for fid in [f.id for f in features]])}"
        )

    # Add feature ids to used ids if necessary
    if used_ids is None:
        used_ids = []
    for feature in features:
        if feature.id not in used_ids:
            used_ids.append(feature.id)

    # Set up c-constants
    cdef cConstants* cconstants = constants.to_c()

    # Initialize grid (if necessary)
    if grid is None:
        # SR_TODO figure out appropriate n_slots (no. neighbors maybe?)
        grid = Grid(constants, alloc_tables=True, n_slots=1)
    else:
        raise Exception("memory leak with external Grid")
    grid.reset_tables()
    cdef cGrid* cgrid = grid.to_c()

    # Turn features into cregions
    cdef int n_features = len(features)
    cdef cRegions cregions = cregions_create(n_features)
    cdef bint ignore_missing_neighbors = True
    features_to_cregions(
        features,
        n_features,
        &cregions,
        cregion_conf_default(),
        ignore_missing_neighbors,
        cgrid,
        cconstants,
    )

    # Create cregions for grown features
    cdef cRegions cregions_grown = cregions_create(n_features)

    # Grow regions
    cfeatures_grow_core(n, &cregions, &cregions_grown, cgrid)

    # SR_TODO also expand values (dummy values for new pixels)

    cdef list features_grown
    cdef np.ndarray[np.float32_t, ndim=1] values_orig, values
    cdef np.ndarray[np.int32_t, ndim=2] pixels_orig
    cdef list shells_orig, shells
    cdef list holes_orig, holes
    cdef np.ndarray[np.int32_t, ndim=2] shell, hole
    if inplace:
        for i, feature in enumerate(features):
            cregion = cregions_grown.regions[i]

            feature._cache = {}

            if retain_orig:
                # Store original values, pixels, etc.
                values_orig = feature.values
                pixels_orig = feature.pixels
                shells_orig = feature.shells
                holes_orig = feature.holes
                feature.properties["feature_orig"] = {}
                feature.properties["feature_orig"]["values"] = values_orig
                feature.properties["feature_orig"]["pixels"] = pixels_orig
                feature.properties["feature_orig"]["shells"] = shells_orig
                feature.properties["feature_orig"]["holes"]  = holes_orig

            # Transfer values, pixels, etc. from cregions
            values = np.full(cregion.pixels_n, -1, dtype=np.float32)
            # + for j in range(cregion.pixels_max):
            # +     if cregion.pixels[j] is not NULL:
            # +         values[j] = cregion.pixels[j].v
            feature.set_values(values)
            feature.set_pixels(cpixel2arr(cregion.pixels, cregion.pixels_n))
            shells = []
            for j in range(cregion.shells_n):
                shell = cpixel2arr(cregion.shells[j], cregion.shell_n[j])
                shells.append(shell)
            feature.set_shells(shells)
            holes = []
            for j in range(cregion.holes_n):
                hole = cpixel2arr(cregion.holes[j], cregion.hole_n[j])
                holes.append(hole)
            feature.set_holes(holes)
            feature.set_cregion(cregion)
    else:
        # Turn cregions into Feature objects
        ignore_missing_neighbors = False
        features_grown = cregions_create_features(
            &cregions_grown,
            base_id,
            ignore_missing_neighbors,
            cgrid,
            cconstants,
            used_ids=used_ids
        )
        features_reset_cregion(features_grown, warn=False)

    # Clean up memory
    cregions_cleanup(&cregions, cleanup_regions=True)
    features_reset_cregion(features, warn=False)

    if not inplace:
        return features_grown


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_grow
# :call: v --- calling ---
# :call: v stormtrack::core::identification::assert_no_unambiguously_assigned_pixels
# :call: v stormtrack::core::identification::regiongrow_advance_boundary
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::tables::pixel_region_table_alloc_pixels
# :call: v stormtrack::core::tables::pixel_region_table_init_regions
# :call: v stormtrack::core::tables::pixel_region_table_reset_region
# :call: v stormtrack::core::tables::pixel_status_table_init_feature
# :call: v stormtrack::core::tables::pixel_status_table_reset_feature
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregion::cregion_insert_pixels_coords
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: v stormtrack::core::cregions::cregions_find_connected
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::grid::grid_create_cregion
cdef void cfeatures_grow_core(
    int ngrow, cRegions* cregions_orig, cRegions* cregions_grown, cGrid* grid,
) except *:
    cdef bint debug = False
    if debug:
        log.debug("< cfeatures_grow_core")
    cdef int n_regions = cregions_orig.n
    cdef int i
    cdef int j
    cdef int k
    cdef int i_region
    cdef int neighbor_rank
    cdef cRegion* cregion
    cdef cPixel* cpixel
    cdef np.int32_t x
    cdef np.int32_t y

    # Initialize dummy feature covering the whole domain to grow into
    if debug: log.debug("  initialize dummy region for domain")
    cdef cRegion cfeature
    cdef cRegionConf cregion_conf = cRegionConf(
        connected_max=0,
        pixels_max=grid.constants.nx*grid.constants.ny,
        shells_max=1,
        shell_max=2*grid.constants.nx + 2*grid.constants.ny,
        holes_max=0,
        hole_max=0,
    )
    cregion_init(&cfeature, cregion_conf, cregion_get_unique_id())
    cdef np.ndarray[np.int32_t, ndim=2] grid_pixels = np.array(
        np.meshgrid(np.arange(grid.constants.nx), np.arange(grid.constants.ny)),
        np.int32,
    ).reshape([2, grid.constants.nx*grid.constants.ny]).T
    cregion_insert_pixels_coords(
        &cfeature, grid.pixels, grid_pixels, link_region=True, unlink_pixels=False,
    )

    # SR_TODO Move all the initialization into a function (same for cleanup)
    # SR_TODO Thus it will be visible in profiling and easy to pull up
    if debug:
        log.debug("  initialize tables etc.")

    # 'Activate' new subregions
    for i in range(n_regions):
        cregions_link_region(
            cregions_grown, grid_create_cregion(grid), cleanup=False, unlink_pixels=False,
        )

    # Initialize pixel status table for feature
    pixel_status_table_init_feature(grid.pixel_status_table, &cfeature, cregions_orig)

    # Allocate slots in pixel region table for feature pixels
    pixel_region_table_alloc_pixels(grid.pixel_region_table, n_regions, &cfeature)

    # Initialize pixel region table with seed regions
    cdef int n_regions_max = 3
    pixel_region_table_init_regions(
        grid.pixel_region_table, cregions_orig, cregions_grown, n_regions_max,
    )

    # Initialize regions for boundary pixels
    cdef cRegions features_iteration_seeds = cregions_create(cregions_orig.n)

    # Add seed pixels to regions and initial seeds
    if debug:
        log.debug("  prepare seed pixels etc.")
    cdef int n
    cdef int l
    cdef cPixel** pixels_tmp
    cdef cRegion* cregion_seeds
    cdef cRegion* cregion_iter
    for i in range(n_regions):
        k = cregions_grown.n - n_regions + i
        cregion = cregions_grown.regions[k]
        cregion_seeds = cregions_orig.regions[i]
        cregion_iter = grid_create_cregion(grid)
        cregions_link_region(
            &features_iteration_seeds, cregion_iter, cleanup=False, unlink_pixels=False,
        )
        n = cregion_seeds.pixels_n
        pixels_tmp = <cPixel**>malloc(n*sizeof(cPixel*))
        l = 0
        for j in range(cregion_seeds.pixels_max):
            if cregion_seeds.pixels[j] is NULL:
                continue
            pixels_tmp[l] = cregion_seeds.pixels[j]
            l += 1
        for j in range(n):
            cpixel = pixels_tmp[j]
            cregion_insert_pixel(cregion, cpixel, link_region=True, unlink_pixel=False)
            cregion_insert_pixel(
                cregion_iter, cpixel, link_region=False, unlink_pixel=False,
            )
        free(pixels_tmp)

    # Grow iteratively one pixel wide boundaries at a time
    # until all feature pixels have been assigned to regions
    if debug:
        log.debug("  grow regions")
    cdef iter_i
    cdef iter_max=ngrow
    for iter_i in range(iter_max):
        if debug:
            log.debug(f"\n  <<<<<<<<<< ITER {iter_i} <<<<<<<<<<<<")

        # Grow region by one pixel along boundary
        regiongrow_advance_boundary(
            &features_iteration_seeds, cregions_grown, n_regions, grid,
        )
        if debug:
            assert_no_unambiguously_assigned_pixels(&cfeature, grid.pixel_status_table)

        # If the regions have not grown any further, we're done early
        for i in range(features_iteration_seeds.n):
            if features_iteration_seeds.regions[i].pixels_n > 0:
                break
        else:
            if debug:
                log.debug(
                    f"  DONE in {iter_i + 1} iterations "
                    f"(features_iteration_seeds is NULL)"
                )
            break

    # Identify boundaries and connected regions
    if debug:
        log.debug("  post-process grown features")
    cregions_determine_boundaries(cregions_grown, grid)
    cdef bint reset_existing = False # SR_TODO True or False?
    cregions_find_connected(cregions_grown, reset_existing, &grid.constants)

    if debug:
        log.debug("  clean up")

    # Reset pixel status table for feature pixels
    # (Use new subfeatures because the pixels have been removed
    # from the original cfeature in the splitting process)
    for i in range(cregions_grown.n):
        pixel_status_table_reset_feature(
            grid.pixel_status_table, cregions_grown.regions[i],
        )

    # Clean up slots in pixel region table for all feature pixels
    pixel_region_table_reset_region(grid.pixel_region_table, &cfeature)

    # Clean up rest
    cregion_cleanup(&cfeature, unlink_pixels=True, reset_connected=True)
    cregions_cleanup(&features_iteration_seeds, cleanup_regions=True)

    if debug: log.debug("> cfeatures_grow_core")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
cpdef void _replace_feature_associations(
    Feature feature, list seed_features, list subfeatures,
) except *:
    """If seed features are assoc'd with input feature, replace references.

    That is, the seed features should no longer be associated with the
    original feature, but with the respective new subfeature

    """
    cdef bint debug = False
    cdef str _name_ = "_replace_feature_associations"
    if debug:
        log.debug(
            f"{_name_}: {feature.id} + "
            f"[{', '.join([str(f.id) for f in seed_features])}] => "
            f"[{', '.join([str(f.id) for f in subfeatures])}]"
        )

    cdef Feature seed_feature
    cdef Feature subfeature
    cdef Feature main_feature_assc
    cdef list main_associates
    cdef list seed_associates
    cdef str name_main
    cdef str name_assc
    cdef np.uint64_t main_feature_assc_id

    # Loop over all assoc'd feature types of the main feature
    for name_assc, main_associates in feature.associates.items():

        # Loop over all associates features of a certain type
        for main_feature_assc_ in main_associates.copy():

            # CASE 1: associates stored as features
            if isinstance(main_feature_assc_, Feature):
                main_feature_assc = main_feature_assc_

                # Loop over all seed features to find the seed feature that
                # corresponds to the current associated feature
                for seed_feature in seed_features:
                    if seed_feature != main_feature_assc:
                        continue

                    # Found match: seed feature equals associated feature
                    # Remove association between main feature and seed feature
                    if debug:
                        log.debug(
                            f"feature {feature.id}: remove {name_assc} associate "
                            f"{seed_feature.id}"
                        )
                    main_associates.remove(seed_feature)

                    # Loop over all assoc'd feature types of the seed feature
                    # to find the group that contains the associated feature
                    for name_main, seed_associates in (seed_feature.associates.items()):
                        if feature not in seed_associates:
                            continue

                        # Found match; now remove associated feature from
                        # associated feature group of seed feature
                        if debug:
                            log.debug(
                                f"feature {seed_feature.id}: remove {name_main} "
                                f"associate {feature.id}"
                            )
                        seed_associates.remove(feature)

                        # Loop over all new subfeatures to find the one which
                        # overlaps with the current seed feature and link them
                        for subfeature in subfeatures:
                            if subfeature.overlaps(seed_feature):
                                if debug:
                                    log.debug(
                                        f"feature {seed_feature.id}: add {name_main} "
                                        f"associate {subfeature.id}"
                                    )
                                seed_associates.append(subfeature)
                                if name_assc not in subfeature.associates:
                                    subfeature.associates[name_assc] = []
                                if debug:
                                    log.debug(
                                        f"feature {subfeature.id}: add {name_assc} "
                                        f"associate {seed_feature.id}"
                                    )
                                subfeature.associates[name_assc].append(seed_feature)
                                break
                        else:
                            raise Exception("ERROR0")
                        break
                    break

            # CASE 2: associates stored as feature ids
            else:
                main_feature_assc_id = main_feature_assc_

                # Loop over all seed features to find the seed feature that
                # corresponds to the current associated feature
                for seed_feature in seed_features:
                    if feature.id != main_feature_assc_id:
                        continue

                    # Found match: seed feature equals associated feature
                    # Now remove the associations between the two
                    if debug:
                        log.debug(
                            f"feature {feature.id}: remove {name_assc} associate "
                            f"{seed_feature.id}"
                        )
                    main_associates.remove(seed_feature.id)

                    # Loop over all assoc'd feature types of the seed feature
                    # to find the group that contains the associated feature
                    for name_main, seed_associates in (
                            seed_feature.associates.items()):
                        if main_feature_assc_id not in seed_associates:
                            continue

                        # Found match; now remove associated feature from
                        # associated feature group of seed feature
                        if debug:
                            log.debug(
                                f"feature {seed_feature.id}: remove {name_main} "
                                f"associate {feature.id}"
                            )
                        seed_associates.remove(feature.id)

                        # Loop over all new subfeatures to find the one which
                        # overlaps with the current seed feature and link them
                        for subfeature in subfeatures:
                            if subfeature.overlaps(seed_feature):
                                if debug:
                                    log.debug(
                                        f"feature {seed_feature.id}: add {name_main} "
                                        f"associate {subfeature.id}"
                                    )
                                seed_associates.append(subfeature.id)
                                if name_assc not in subfeature.associates:
                                    subfeature.associates[name_assc] = []
                                if debug:
                                    log.debug(
                                        f"feature {subfeature.id}: add {name_assc} "
                                        f"associate {seed_feature.id}"
                                    )
                                subfeature.associates[name_assc].append(seed_feature.id)
                                break
                        else:
                            raise Exception("ERROR0")
                        break
                    break


# :call: > --- callers ---
# :call: > stormtrack::core::identification::identify_features
# :call: > test_stormtrack::test_core::test_features::test_split_regiongrow::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::csplit_regiongrow_levels
# :call: v stormtrack::core::constants::default_constants
def split_regiongrow_levels(
    features,
    levels,
    *,
    nx=0,
    ny=0,
    base_id=0,
    minsize=1,
    maxsize=-1,
    seed_minsize=0,
    seed_min_strength=0,
    filter_mask=None, # SR_TODO add min overlap with filter regions
    filter_min_overlap=-1.0,
    constants=None,
):
    levels = np.asarray(levels, dtype=np.float32)

    if constants is None:
        if nx == 0 or ny == 0:
            raise Exception("must pass either (nx, ny) or constants")
        constants = default_constants(nx=nx, ny=ny)
    else:
        nx = constants.nx
        ny = constants.ny

    # filter: features which overlap with mask are removed
    if filter_mask is None:
        apply_mask_filter = False
        filter_mask = np.zeros([nx, ny], dtype=np.uint8)
    else:
        apply_mask_filter = True
        filter_mask = np.asarray(filter_mask, dtype=np.uint8)
        if filter_mask.shape != (nx, ny):
            raise ValueError(
                f"filter_mask: invalid shape: {filter_mask.shape} != ({nx}, {ny})"
            )

    cdef list subfeatures = csplit_regiongrow_levels(
        features=features,
        levels=levels,
        nx=nx,
        ny=ny,
        base_id=base_id,
        minsize=minsize,
        maxsize=maxsize,
        seed_minsize=seed_minsize,
        seed_min_strength=seed_min_strength,
        apply_mask_filter=apply_mask_filter,
        filter_mask=filter_mask,
        filter_min_overlap=filter_min_overlap,
        constants=constants,
    )
    return subfeatures


# :call: > --- callers ---
# :call: > stormtrack::core::identification::split_regiongrow_levels
# :call: v --- calling ---
# :call: v stormtrack::core::identification::cregions_create_features
# :call: v stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: v stormtrack::core::identification::features_reset_cregion
# :call: v stormtrack::core::identification::features_to_cregions
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::pixel_done_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v stormtrack::core::tables::pixel_region_table_alloc_pixels
# :call: v stormtrack::core::tables::pixel_region_table_reset_region
# :call: v stormtrack::core::tables::pixel_status_table_alloc
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregion::cregion_overlap_n_mask
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: v stormtrack::core::cregions::cregions_find_connected
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::grid::grid_cleanup
# :call: v stormtrack::core::grid::grid_create
# :call: v stormtrack::core::grid::grid_create_cregion
cdef list csplit_regiongrow_levels(
    list features,
    np.ndarray[np.float32_t, ndim=1] levels,
    np.int32_t nx,
    np.int32_t ny,
    np.uint64_t base_id,
    int minsize,
    int maxsize,
    int seed_minsize,
    np.float32_t seed_min_strength,
    bint apply_mask_filter,
    np.ndarray[np.uint8_t, ndim=2]   filter_mask,
    float filter_min_overlap,
    Constants constants,
):
    cdef bint debug = False
    if debug:
        log.debug(f"< csplit_regiongrow_levels {levels}")
    cdef int i
    cdef int j
    cdef int k
    cdef int i_feature

    # Set up constants
    cdef cConstants* cconstants = constants.to_c()

    # Initialize grid
    cdef np.ndarray[np.float32_t, ndim=2] field = np.zeros(
        [cconstants.nx, cconstants.ny], dtype=np.float32,
    )
    cdef cGrid grid = grid_create(field, cconstants[0])

    # Allocate/initialize lookup tables
    neighbor_link_stat_table_alloc(&grid.neighbor_link_stat_table, cconstants)
    pixel_done_table_alloc(&grid.pixel_done_table, cconstants)
    pixel_status_table_alloc(&grid.pixel_status_table, cconstants)
    pixel_region_table_alloc_grid(&grid.pixel_region_table, cconstants)

    # Turn features into cregions
    cdef int nmax_cregions = 10
    cdef int n_features = len(features)
    cdef cRegions cfeatures = cregions_create(nmax_cregions)
    cdef bint ignore_missing_neighbors = False
    features_to_cregions(
        features,
        n_features,
        &cfeatures,
        cregion_conf_default(),
        ignore_missing_neighbors,
        &grid,
        cconstants,
    )

    # Initialize cregions for subregions and seeds
    cdef cRegions subfeatures_tmp = cregions_create(nmax_cregions)
    cdef cRegions subfeatures_final = cregions_create(nmax_cregions)

    cdef int n
    cdef cPixel** pixels_tmp

    # Iterate over the input features
    cdef float ratio
    cdef np.float32_t level
    cdef cPixel* cpixel
    cdef cRegion* cregion
    cdef cRegion* cfeature_sub
    cdef int nmax_seeds = 20
    for i_feature in range(cfeatures.n):
        cfeature = cfeatures.regions[i_feature]

        # Start with full feature
        subfeatures_tmp.regions[0] = cfeature
        subfeatures_tmp.n = 1

        # Allocate slots in pixel region table for all feature pixels
        pixel_region_table_alloc_pixels(grid.pixel_region_table, nmax_seeds, cfeature)

        # Split features
        csplit_regiongrow_levels_core(
            cfeature,
            &subfeatures_tmp,
            &subfeatures_final,
            levels,
            &grid,
            &nmax_seeds,
            seed_minsize,
            seed_min_strength,
        )

        # Clean up slots in pixel region table for all feature pixels
        pixel_region_table_reset_region(grid.pixel_region_table, cfeature)

        # SR_TODO move into function
        # Finish regions
        for i in range(subfeatures_tmp.n):

            # Check feature size
            if minsize > 1:
                cregion = subfeatures_tmp.regions[i]
                if cregion.pixels_n < minsize:
                    if debug:
                        log.debug(
                            f"remove [{cregion.id}] (n={cregion.pixels_n}): "
                            f"too small (< {minsize})"
                        )
                    continue
            if maxsize > 0:
                cregion = subfeatures_tmp.regions[i]
                if cregion.pixels_n > maxsize:
                    if debug:
                        log.debug(
                            f"remove [{cregion.id}] (n={cregion.pixels_n}): "
                            f"too big (> {maxsize})"
                        )
                    continue

            # Filter features by overlap with mask
            # Features which overlap with mask are removed
            # SR_TODO implement threshold for overlap (absolute or relative)
            if apply_mask_filter:
                cregion = subfeatures_tmp.regions[i]
                n = cregion_overlap_n_mask(cregion, filter_mask)
                if n > 0:
                    ratio = <float>n/<float>cregion.pixels_n
                    if ratio > filter_min_overlap:
                        if debug:
                            log.debug(
                                f"remove [{cregion.id}] (n={cregion.pixels_n}): "
                                f"overlaps with mask: {n} ({ratio})"
                            )
                        continue

            # SR_TODO Figure out if this can be removed!
            # Not exactly sure why I'm removing gaps in pixels array here...
            cregion = grid_create_cregion(&grid)
            cregions_link_region(
                &subfeatures_final, cregion, cleanup=False, unlink_pixels=False,
            )
            n = subfeatures_tmp.regions[i].pixels_n
            pixels_tmp = <cPixel**>malloc(n*sizeof(cPixel*))
            k = 0
            for j in range(subfeatures_tmp.regions[i].pixels_max):
                if subfeatures_tmp.regions[i].pixels[j] is NULL:
                    continue
                pixels_tmp[k] = subfeatures_tmp.regions[i].pixels[j]
                k += 1
            for j in range(n):
                cpixel = pixels_tmp[j]
                cregion_insert_pixel(
                    cregion, cpixel, link_region=True, unlink_pixel=False,
                )
            free(pixels_tmp)
        if debug:
            log.debug(f"subfeatures_final.n={subfeatures_final.n}")

    # Identify boundaries and connected regions
    cregions_determine_boundaries(&subfeatures_final, &grid)
    cdef bint reset_existing = False # True or False?
    cregions_find_connected(&subfeatures_final, reset_existing, cconstants)

    # Turn cregion into Feature objects
    ignore_missing_neighbors = False
    cdef list subfeatures = cregions_create_features(
        &subfeatures_final, base_id, ignore_missing_neighbors, &grid, cconstants,
    )

    # Clean up cregions
    cregions_cleanup(&subfeatures_tmp, cleanup_regions=True)
    cregions_cleanup(&subfeatures_final, cleanup_regions=True)

    # Clean up grid
    grid_cleanup(&grid)
    features_reset_cregion(features, warn=False)
    features_reset_cregion(subfeatures, warn=False)

    if debug:
        log.debug(f"> csplit_regiongrow_levels {len(subfeatures)}")

    return subfeatures


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: v --- calling ---
# :call: v stormtrack::core::identification::csplit_regiongrow_core
# :call: v stormtrack::core::identification::extract_subregions_level
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::tables::neighbor_link_stat_table_reset_pixels
# :call: v stormtrack::core::tables::pixel_region_table_grow
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::cregions::cregions_reset
# :call: v stormtrack::core::grid::grid_create_cregion
cdef void csplit_regiongrow_levels_core(
    cRegion* cfeature,
    cRegions* subfeatures_tmp,
    cRegions* subfeatures_final,
    np.ndarray[np.float32_t, ndim=1] levels,
    cGrid* grid,
    int* nmax_seeds,
    int seed_minsize,
    np.float32_t seed_min_strength,
) except *:
    # print("< csplit_regiongrow_levels_core")
    cdef int i_level
    cdef int i_region
    cdef np.float32_t level
    cdef cRegion* cregion
    cdef cPixel* cpixel
    cdef int nmax_cregions = 10
    cdef cRegions subfeatures_seeds = cregions_create(nmax_cregions)
    cdef cRegions subfeatures_new = cregions_create(nmax_cregions)
    cdef int i
    cdef int j
    cdef int k
    cdef int n
    cdef cPixel** pixels_tmp

    # Iterate over the levels
    cdef int n_levels = levels.shape[0]
    for i_level in range(n_levels):
        level = levels[i_level]
        # print(f"\n[{i_level}] LEVEL {level}")

        # Loop over all previously split subregions
        cregions_reset(&subfeatures_new)
        for i_region in range(subfeatures_tmp.n):
            cfeature_sub = subfeatures_tmp.regions[i_region]

            # Reset neighbor link status table
            neighbor_link_stat_table_reset_pixels(
                grid.neighbor_link_stat_table, cfeature, grid.constants.n_neighbors_max,
            )

            # Build seed features
            # print(f"[{i_region}] FIND SEED FEATURES (LEVEL {level})")
            cregions_reset(&subfeatures_seeds)
            extract_subregions_level(
                cfeature_sub,
                &subfeatures_seeds,
                level,
                grid,
                seed_minsize,
                seed_min_strength,
            )
            # print(f"subfeatures_seeds.n={subfeatures_seeds.n}")

            # If no seed has been found, this subfeature is finished
            if subfeatures_seeds.n == 0:
                # print("region finished!")
                cregion = grid_create_cregion(grid)
                cregions_link_region(
                    subfeatures_final, cregion, cleanup=False, unlink_pixels=False,
                )
                n = cfeature_sub.pixels_n
                # SR_TODO move malloc/free out of loop
                pixels_tmp = <cPixel**>malloc(n*sizeof(cPixel*))
                j = 0
                for i in range(cfeature_sub.pixels_max):
                    if cfeature_sub.pixels[i] is NULL:
                        continue
                    pixels_tmp[j] = cfeature_sub.pixels[i]
                    j += 1
                for i in range(n):
                    cpixel = pixels_tmp[i]
                    # SR_TODO check whether unlink_pixel is necessary
                    cregion_insert_pixel(
                        cregion, cpixel, link_region=True, unlink_pixel=True,
                    )
                free(pixels_tmp)
                continue

            # Grow the pixel region table if necessary
            if subfeatures_seeds.n > nmax_seeds[0]:
                pixel_region_table_grow(
                    grid.pixel_region_table, cfeature, subfeatures_seeds.n,
                )
                nmax_seeds[0] = subfeatures_seeds.n

            # Split subfeature from seeds
            # print(f"[{i_region}] SPLIT SUBFEATURE")

            csplit_regiongrow_core(
                cfeature_sub,
                &subfeatures_seeds,
                &subfeatures_new,
                subfeatures_seeds.n,
                grid,
            )
            # print(f"subfeatures_new.n={subfeatures_new.n}")

        # Collect new subfeatures in 'temporary storage'
        cregions_reset(subfeatures_tmp)
        for i in range(subfeatures_new.n):
            cregion = grid_create_cregion(grid)
            cregions_link_region(
                subfeatures_tmp, cregion, cleanup=False, unlink_pixels=False,
            )
            n = subfeatures_new.regions[i].pixels_n
            pixels_tmp = <cPixel**>malloc(n*sizeof(cPixel*))
            k = 0
            for j in range(subfeatures_new.regions[i].pixels_max):
                if subfeatures_new.regions[i].pixels[j] is NULL:
                    continue
                pixels_tmp[k] = subfeatures_new.regions[i].pixels[j]
                k += 1
            for j in range(n):
                cpixel = pixels_tmp[j]
                cregion_insert_pixel(
                    cregion, cpixel, link_region=True, unlink_pixel=False,
                )
            free(pixels_tmp)

    # Cleanup
    cregions_cleanup(&subfeatures_seeds, cleanup_regions=True)
    cregions_cleanup(&subfeatures_new, cleanup_regions=True)
    # print("> csplit_regiongrow_levels_core")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: v --- calling ---
# :call: v stormtrack::core::identification::collect_adjacent_pixels
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::tables::pixel_done_table_init
# :call: v stormtrack::core::tables::pixel_done_table_reset
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregion::cregion_reset
# :call: v stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::grid::grid_create_cregion
cdef void extract_subregions_level(
    cRegion* cregion,
    cRegions* cregions_sub,
    np.float32_t level,
    cGrid* grid,
    int subregion_minsize,
    np.float32_t subregion_min_strength,
) except *:
    cdef bint debug = False
    if debug:
        log.debug(f"< extract_subregions_level {cregion.id} {level}")

    # Initialize lookup table for the pixels
    pixel_done_table_init(grid.pixel_done_table, cregion, level)

    # Extract pixels (region growing)
    cdef int i_pixel
    cdef int i_pixel_start
    cdef int i
    cdef int j
    cdef int k
    cdef int n
    cdef int i_neighbor
    cdef cPixel* cpixel
    cdef np.float32_t vmax
    cdef cPixel* neighbor
    cdef cRegion cregion_tmp
    cdef cPixel** pixels_tmp
    cregion_init(&cregion_tmp, cregion_conf_default(), cregion_get_unique_id())
    n = cregion.pixels_n
    pixels_tmp = <cPixel**>malloc(n*sizeof(cPixel))
    j = 0
    for i in range(cregion.pixels_max):
        if cregion.pixels[i] is NULL:
            continue
        pixels_tmp[j] = cregion.pixels[i]
        j += 1
    for i_pixel_start in range(n):
        cpixel = pixels_tmp[i_pixel_start]
        if grid.pixel_done_table[cpixel.x][cpixel.y]:
            continue

        # Collect all adjacent pixels
        collect_adjacent_pixels(cpixel, &cregion_tmp, grid)
        valid_subregion = True

        # Check size
        if cregion_tmp.pixels_n < subregion_minsize:
            valid_subregion = False

        # Check min. strength
        if valid_subregion:
            vmax = 0.0
            for i_pixel in range(cregion_tmp.pixels_max):
                if cregion_tmp.pixels[i_pixel] is NULL:
                    continue
                if cregion_tmp.pixels[i_pixel].v > vmax:
                    vmax = cregion_tmp.pixels[i_pixel].v
            if vmax < subregion_min_strength:
                valid_subregion = False

        # Reset tmp region; store subregion if all requirements met
        if valid_subregion:
            cregion_sub = grid_create_cregion(grid)
            cregions_link_region(
                cregions_sub, cregion_sub, cleanup=False, unlink_pixels=False,
            )
        for i_pixel in range(cregion_tmp.pixels_istart,
                cregion_tmp.pixels_iend):
            if cregion_tmp.pixels[i_pixel] is not NULL:
                if valid_subregion:
                    cregion_insert_pixel(
                        cregion_sub,
                        cregion_tmp.pixels[i_pixel],
                        link_region=True,
                        unlink_pixel=False,
                    )
        # SR_TODO is reset_connected necessary?
        cregion_reset(&cregion_tmp, unlink_pixels=False, reset_connected=False)
    free(pixels_tmp)

    # DBG_BLOCK <
    if debug:
        log.debug("PIXEL REGIONS:")
        for i in range(cregion.pixels_max):
            cpixel = cregion.pixels[i]
            if cpixel is NULL:
                continue
            if cpixel.region is NULL:
                log.debug(f"[{-1}] ({cpixel.x}, {cpixel.y})")
            else:
                log.debug(f"[{cpixel.region.id}] ({cpixel.x}, {cpixel.y})")
        log.debug(
            cregions_sub.n, cregions_sub.regions[0].id, cregions_sub.regions[1].id,
        )
        log.debug("SUBREGIONS:")
        for i in range(cregions_sub.n):
            for j in range(cregions_sub.regions[i].pixels_max):
                if cregions_sub.regions[i].pixels[j] is not NULL:
                    log.debug(
                        f"[{cregions_sub.regions[i].id}] "
                        f"({cregions_sub.regions[i].pixels[j].x}, "
                        f"{cregions_sub.regions[i].pixels[j].y})"
                    )
    # DBG_BLOCK >

    # SR_TODO unlink_pixels or not?
    # SR_TODO is unlink_pixel and/or reset_connected necessary?
    cregion_reset(&cregion_tmp, unlink_pixels=False, reset_connected=True)
    cregions_determine_boundaries(cregions_sub, grid)

    # DBG_BLOCK <
    if debug:
        for i in range(cregions_sub.n):
            log.debug(
                f"[{cregions_sub.regions[i].id}] PIXEL "
                f"{cregions_sub.regions[i].pixels_n}"
            )
            for j in range(cregions_sub.regions[i].pixels_max):
                if cregions_sub.regions[i].pixels[j] is not NULL:
                    log.debug(
                        f"[{cregions_sub.regions[i].id}] PIXEL "
                        f"({cregions_sub.regions[i].pixels[j].x}, "
                        f"{cregions_sub.regions[i].pixels[j].y})"
                    )
            for j in range(cregions_sub.regions[i].shells_n):
                log.debug(
                    f"[{cregions_sub.regions[i].id}] SHELL({j}) "
                    f"{cregions_sub.regions[i].shells_n}"
                )
                for k in range(cregions_sub.regions[i].shells_n):
                    log.debug(
                        f"[{cregions_sub.regions[i].id}] SHELL({j}) "
                        f"({cregions_sub.regions[i].shells[j][k].x}, "
                        f"{cregions_sub.regions[i].shells[j][k].x})"
                    )
    # DBG_BLOCK >

    # Reset lookup table for the pixels
    pixel_done_table_reset(grid.pixel_done_table, cregion)

    if debug:
        log.debug(f"> extract_subregions_level {cregions_sub.n}")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregion::cregion_remove_pixel
cdef void collect_adjacent_pixels(
    cPixel* cpixel,
    cRegion* cregion,
    cGrid* grid,
) except *:
    # print(f"< collect_adjacent_pixels ({cpixel.x}, {cpixel.y})")
    grid.pixel_done_table[cpixel.x][cpixel.y] = True
    cdef cRegion pixels_todo
    cregion_init(&pixels_todo, cregion_conf_default(), cregion_get_unique_id())
    # print(f"pixels_todo.id == {pixels_todo.id}")
    cdef int i
    cdef int i_neighbor
    cdef cPixel* neighbor
    cdef cPixel* current_cpixel = cpixel
    cdef int iter_i
    cdef int iter_max = 1000000
    for iter_i in range(iter_max):
        for i in range(pixels_todo.pixels_max):
            if pixels_todo.pixels[i] is NULL:
                continue

        # Add the current pixel to the region
        cregion_insert_pixel(
            cregion, current_cpixel, link_region=False, unlink_pixel=False,
        )

        # Add all neighbors to queue
        for i_neighbor in range(grid.constants.n_neighbors_max):
            neighbor = current_cpixel.neighbors[i_neighbor]
            if neighbor is NULL or grid.pixel_done_table[neighbor.x][neighbor.y]:
                continue
            cregion_insert_pixel(
                &pixels_todo, neighbor, link_region=False, unlink_pixel=False,
            )
            grid.pixel_done_table[neighbor.x][neighbor.y] = True

        # Select the next pixel in the queue
        for i in range(pixels_todo.pixels_max):
            if pixels_todo.pixels[i] is NULL:
                continue
            current_cpixel = pixels_todo.pixels[i]
            cregion_remove_pixel(&pixels_todo, current_cpixel)
            break
        else:
            break
    else:
        raise Exception(f"identify_regions: loop timed out (nmax={iter_max})")
    # SR_TODO is reset_connected necessary?
    cregion_cleanup(&pixels_todo, unlink_pixels=False, reset_connected=True)
    # print("> collect_adjacent_pixels")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: v --- calling ---
# :call: v stormtrack::core::identification::assert_no_unambiguously_assigned_pixels
# :call: v stormtrack::core::identification::regiongrow_advance_boundary
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::tables::pixel_region_table_alloc_pixels
# :call: v stormtrack::core::tables::pixel_region_table_init_regions
# :call: v stormtrack::core::tables::pixel_region_table_reset_region
# :call: v stormtrack::core::tables::pixel_status_table_init_feature
# :call: v stormtrack::core::tables::pixel_status_table_reset_feature
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: v stormtrack::core::cregions::cregions_find_connected
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::grid::grid_create_cregion
cdef void csplit_regiongrow_core(
    cRegion* cfeature,
    cRegions* cregions_seeds,
    cRegions* subfeature_cregions,
    int n_subregions,
    cGrid* grid,
) except *:
    cdef bint debug = False
    if debug:
        log.debug("< csplit_regiongrow_core")
    cdef int i
    cdef int j
    cdef int k
    cdef int i_region
    cdef int neighbor_rank
    cdef cRegion* cregion
    cdef cPixel* cpixel
    cdef np.int32_t x
    cdef np.int32_t y

    # SR_TODO Move all the initialization into a function (same for cleanup)
    # SR_TODO Thus it will be visible in profiling and easy to pull up

    # 'Activate' new subregions
    for i in range(n_subregions):
        cregions_link_region(
            subfeature_cregions,
            grid_create_cregion(grid),
            cleanup=False,
            unlink_pixels=False,
        )

    # Initialize pixel status table for feature
    pixel_status_table_init_feature(grid.pixel_status_table, cfeature, cregions_seeds)

    # Allocate slots in pixel region table for feature pixels
    pixel_region_table_alloc_pixels(grid.pixel_region_table, n_subregions, cfeature)

    # Initialize pixel region table with seed regions
    cdef int n_regions_max = 3
    pixel_region_table_init_regions(
        grid.pixel_region_table, cregions_seeds, subfeature_cregions, n_regions_max,
    )

    # Initialize regions for boundary pixels
    cdef cRegions subfeatures_iteration_seeds = cregions_create(cregions_seeds.n)

    # Add seed pixels to regions and initial seeds
    cdef int n
    cdef int l
    cdef cPixel** pixels_tmp
    cdef cRegion* cregion_seeds
    cdef cRegion* cregion_iter
    for i in range(n_subregions):
        k = subfeature_cregions.n - n_subregions + i
        cregion = subfeature_cregions.regions[k]
        cregion_seeds = cregions_seeds.regions[i]
        cregion_iter = grid_create_cregion(grid)
        cregions_link_region(
            &subfeatures_iteration_seeds,
            cregion_iter,
            cleanup=False,
            unlink_pixels=False,
        )
        n = cregion_seeds.pixels_n
        pixels_tmp = <cPixel**>malloc(n*sizeof(cPixel*))
        l = 0
        for j in range(cregion_seeds.pixels_max):
            if cregion_seeds.pixels[j] is NULL:
                continue
            pixels_tmp[l] = cregion_seeds.pixels[j]
            l += 1
        for j in range(n):
            cpixel = pixels_tmp[j]
            cregion_insert_pixel(cregion, cpixel, link_region=True, unlink_pixel=False)
            cregion_insert_pixel(
                cregion_iter, cpixel, link_region=False, unlink_pixel=False,
            )
        free(pixels_tmp)

    # Grow iteratively one pixel wide boundaries at a time
    # until all feature pixels have been assigned to regions
    cdef iter_i
    cdef iter_max=100000
    for iter_i in range(iter_max):
        if debug:
            log.debug(f"\n<<<<<<<<<< ITER {iter_i} <<<<<<<<<<<<")

        # Grow region by one pixel along boundary
        regiongrow_advance_boundary(
            &subfeatures_iteration_seeds, subfeature_cregions, n_subregions, grid,
        )
        if debug:
            assert_no_unambiguously_assigned_pixels(cfeature, grid.pixel_status_table)

        # If the regions have not grown any further, we're done
        for i in range(subfeatures_iteration_seeds.n):
            if subfeatures_iteration_seeds.regions[i].pixels_n > 0:
                break
        else:
            if debug:
                log.debug(
                    f"DONE in {iter_i + 1} iterations "
                    "(subfeatures_iteration_seeds is NULL)"
                )
            break

    else:
        raise Exception(f"identify_regions: loop timed out (nmax={iter_max})")

    # Identify boundaries and connected regions
    cregions_determine_boundaries(subfeature_cregions, grid)
    cdef bint reset_existing = False # SR_TODO True or False?
    cregions_find_connected(subfeature_cregions, reset_existing, &grid.constants)

    # Reset pixel status table for feature pixels
    # (Use new subfeatures because the pixels have been removed
    # from the original cfeature in the splitting process)
    for i in range(subfeature_cregions.n):
        pixel_status_table_reset_feature(
            grid.pixel_status_table, subfeature_cregions.regions[i],
        )

    # Clean up slots in pixel region table for all feature pixels
    pixel_region_table_reset_region(grid.pixel_region_table, cfeature)

    # Clean up rest
    cregions_cleanup(&subfeatures_iteration_seeds, cleanup_regions=True)

    if debug:
        log.debug("> csplit_regiongrow_core")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
cdef void assert_no_unambiguously_assigned_pixels(
    cRegion* cfeature, np.int8_t** pixel_status_table,
) except *:
    """Check that all assigned pixels are assigned unambiguously"""
    cdef int i
    cdef np.int32_t x
    cdef np.int32_t y
    for i in range(cfeature.pixels_max):
        if cfeature.pixels[i] is NULL:
            continue
        x = cfeature.pixels[i].x
        y = cfeature.pixels[i].y
        if pixel_status_table[x][y] == 2:
            raise Exception(f"pixel_status_table[{x}][{y}] == 2")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: v --- calling ---
# :call: v stormtrack::core::identification::regiongrow_assign_pixel
# :call: v stormtrack::core::identification::regiongrow_resolve_multi_assignments
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
# :call: v stormtrack::core::cregion::cregion_insert_pixel
cdef void regiongrow_advance_boundary(
    cRegions* subfeatures_iteration_seeds,
    cRegions* subfeature_cregions,
    int n_subregions,
    cGrid* grid,
) except *:
    cdef bint debug = False
    cdef bint superdebug = False
    if debug:
        log.debug("< regiongrow_advance_boundary")
    cdef int i
    cdef int j
    cdef int k
    cdef int i_region
    cdef int i_pixel
    cdef int i_neighbor
    cdef int i_subregion
    cdef int neighbor_rank
    cdef cRegion* cregion
    cdef cPixel* cpixel
    cdef cPixel* neighbor
    cdef bint newly_assigned
    cdef np.int32_t x
    cdef np.int32_t y

    # DBG_BLOCK <
    if debug:
        if superdebug: log.debug("========================================")
        log.debug("seed pixels of {} regions:".format(subfeatures_iteration_seeds.n))
        if superdebug: log.debug("========================================")
        for i_region in range(subfeatures_iteration_seeds.n):
            cregion = subfeatures_iteration_seeds.regions[i_region]
            log.debug(f"cregion {cregion.id}: {cregion.pixels_n}")
            if superdebug:
                for i_pixel in range(cregion.pixels_max):
                    if cregion.pixels[i_pixel] is NULL:
                        continue
                    log.debug(
                        f" [{i_region}] "
                        f"({cregion.pixels[i_pixel].x}, {cregion.pixels[i_pixel].y})"
                    )
                log.debug("")
    # SR_BLOCK >

    # SR_TODO Yet another case where cRegion is not really appropriate
    # SR_TODO but still used because it's a flexible cPixel container!
    # SR_TODO I.e. yet another cry for a more general-purpose cPixels struct!
    # Initialize pixel container for multi-assigned pixels
    cdef cRegion cregion_multi_assigned
    cregion_init(
        &cregion_multi_assigned, cregion_conf_default(), cregion_get_unique_id(),
    )
    cdef cRegion newly_assigned_pixels
    cregion_init(
        &newly_assigned_pixels, cregion_conf_default(), cregion_get_unique_id(),
    )

    # DBG_BLOCK <
    if debug:
        if superdebug:
            log.debug("========================================")
        log.debug(f"assign boundary pixels to {subfeatures_iteration_seeds.n} regions")
        if superdebug:
            log.debug("========================================")
    # DBG_BLOCK >

    cdef cRegion* cregion_seeds
    cdef cRegion* cregion_target

    # Loop over the seed pixels of the cregions
    for i_region in range(subfeatures_iteration_seeds.n):
        cregion_seeds = subfeatures_iteration_seeds.regions[i_region]
        cregion_target = subfeature_cregions.regions[
                subfeature_cregions.n - n_subregions + i_region]

        for i_pixel in range(cregion_seeds.pixels_n):
            cpixel = cregion_seeds.pixels[i_pixel]
            if superdebug:
                log.debug(f" [{i_region}] {i_pixel} ({cpixel.x}, {cpixel.y})")

            # Loop over neighbor pixels
            for i_neighbor in range(cpixel.neighbors_max):
                neighbor = cpixel.neighbors[i_neighbor]
                if neighbor is NULL:
                    continue

                # Set neigbor rank:
                # - 0: direct (orthogonal)
                # - 1: indirect (diagnoal)
                with cython.cdivision(True):
                    if i_neighbor%2 == 0:
                        neighbor_rank = 0
                    else:
                        neighbor_rank = 1
                if superdebug:
                    log.debug(
                        f" -> neighbor {i_neighbor}: ({neighbor.x}, {neighbor.y}) "
                        f"({neighbor_rank})"
                    )

                # Assign the pixel
                newly_assigned = regiongrow_assign_pixel(
                    neighbor,
                    neighbor_rank,
                    cregion_target,
                    &cregion_multi_assigned,
                    subfeatures_iteration_seeds.n,
                    grid,
                )
                if newly_assigned:
                    if debug:
                        log.debug(f"newly assigned ({neighbor.x}, {neighbor.y})")
                    cregion_insert_pixel(
                        &newly_assigned_pixels,
                        neighbor,
                        link_region=False,
                        unlink_pixel=False,
                    )

    # DBG_BLOCK <
    if debug:
        if superdebug:
            log.debug("========================================")
        log.debug(f"resolve multi-assigned pixels ({cregion_multi_assigned.pixels_n})")
        if superdebug:
            log.debug("========================================")
    # DBG_BLOCK >

    # Resolve multi-assignments (pixels that might belong to multiple regions)
    regiongrow_resolve_multi_assignments(
        &cregion_multi_assigned, grid, grid.constants.n_neighbors_max, debug = False,
    )
    # SR_TODO is reset_connected necessary?
    cregion_cleanup(
        &cregion_multi_assigned, unlink_pixels=False, reset_connected=True,
    )

    # DBG_BLOCK <
    if debug:
        if superdebug:
            log.debug("========================================")
        log.debug(f"newly assigned pixels ({newly_assigned_pixels.pixels_n}):")
        if superdebug:
            log.debug("========================================")
            for i_region in range(subfeatures_iteration_seeds.n):
                cregion = subfeatures_iteration_seeds.regions[i_region]
                for i_pixel in range(newly_assigned_pixels.pixels_max):
                    cpixel = newly_assigned_pixels.pixels[i_pixel]
                    if cpixel is NULL:
                        continue
                    x = cpixel.x
                    y = cpixel.y
                    if grid.pixel_region_table[x][y].slots[0].region.id == cregion.id:
                        log.debug(f" [{i_region}] ({cpixel.x}, {cpixel.y})")
                log.debug("")
    # DBG_BLOCK >

    # Reset seeds
    for i_region in range(subfeatures_iteration_seeds.n):
        # SR_TODO is reset_connected necessary?
        cregion_cleanup(
            subfeatures_iteration_seeds.regions[i_region],
            unlink_pixels=True,
            reset_connected=True,
        )

    # Store newly assigned pixels
    for i_pixel in range(
        newly_assigned_pixels.pixels_istart, newly_assigned_pixels.pixels_iend
    ):
        cpixel = newly_assigned_pixels.pixels[i_pixel]
        if cpixel is NULL:
            continue

        # Update pixel status table (0: definitely assigned earlier)
        grid.pixel_status_table[cpixel.x][cpixel.y] = 0

        # Add newly assigned pixel to the growing subfeature
        cregion = grid.pixel_region_table[cpixel.x][cpixel.y].slots[0].region
        cregion_insert_pixel(cregion, cpixel, link_region=True, unlink_pixel=False)

        # If the pixel has assignable neighbors, add it to the new seeds
        for i_neighbor in range(grid.constants.n_neighbors_max):
            neighbor = cpixel.neighbors[i_neighbor]
            if neighbor is NULL:
                continue
            if grid.pixel_status_table[neighbor.x][neighbor.y] > 0:
                break
        else:
            continue
        # SR_TMP <
        for i_region in range(subfeature_cregions.n - n_subregions,
                subfeature_cregions.n):
            if subfeature_cregions.regions[i_region].id == cregion.id:
                break
        # SR_TMP >
        cregion = subfeatures_iteration_seeds.regions[i_region]
        cregion_insert_pixel(cregion, cpixel, link_region=False, unlink_pixel=False)
    # SR_TODO is reset_connected necessary?
    cregion_cleanup(&newly_assigned_pixels, unlink_pixels=True, reset_connected=True)

    # DBG_BLOCK <
    if superdebug:
        log.debug("========================================")
        log.debug("pixels of updated growing regions:")
        log.debug("========================================")
        n = subfeature_cregions.n
        log.debug(n, n_subregions)
        for i_region in range(n - n_subregions, n):
            log.debug(i_region, subfeature_cregions.regions[i_region].pixels_n)
            for i_pixel in range(subfeature_cregions.regions[i_region].pixels_max):
                cpixel = subfeature_cregions.regions[i_region].pixels[i_pixel]
                if cpixel is not NULL:
                    log.debug(
                        f" [{i_region}] {subfeature_cregions.regions[i_region].id} "
                        f"({cpixel.x}, {cpixel.y})"
                    )
            log.debug("")
    # DBG_BLOCK >
    if debug:
        log.debug("> regiongrow_advance_boundary")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: v --- calling ---
# :call: v stormtrack::core::identification::resolve_multi_assignment
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionRankSlot
# :call: v stormtrack::core::structs::cRegionRankSlots
# :call: v stormtrack::core::tables::pixel_region_table_insert_region
# :call: v stormtrack::core::tables::pixel_region_table_reset_slots
cdef void regiongrow_resolve_multi_assignments(
    cRegion* cregion_multi_assigned, cGrid* grid, int n_neighbors_max, bint debug,
) except *:
    if debug:
        log.debug("< regiongrow_resolve_multi_assignments")
    cdef int i_pixel
    cdef cPixel* cpixel
    cdef cRegionRankSlot* selected_region_ranked
    cdef np.int32_t x
    cdef np.int32_t y
    cdef int n_selected_regions_max = 3
    cdef cRegionRankSlots selected_regions
    # SR_TODO Use/write generic function to initialize slots
    # SR_TMP <
    selected_regions.slots = <cRegionRankSlot*>malloc(
        n_selected_regions_max*sizeof(cRegionRankSlot),
    )
    selected_regions.max = n_selected_regions_max
    selected_regions.n = 0
    for i in range(selected_regions.max):
        selected_regions.slots[i].region = NULL
        selected_regions.slots[i].rank = -1
    # SR_TMP >

    # Loop over the multi-assigned pixels
    for i_pixel in range(
        cregion_multi_assigned.pixels_istart, cregion_multi_assigned.pixels_iend
    ):
        cpixel = cregion_multi_assigned.pixels[i_pixel]
        if cpixel is NULL:
            continue
        x = cpixel.x
        y = cpixel.y

        if debug:
            log.debug(
                f"({cpixel.x}, {cpixel.y}): "
                f"{grid.pixel_region_table[cpixel.x][cpixel.y].n} regions"
            )
        selected_region_ranked = resolve_multi_assignment(
            cpixel,
            &selected_regions,
            n_selected_regions_max,
            grid,
            n_neighbors_max,
            debug,
        )
        if debug:
            log.debug(f" -> select region {selected_region_ranked.region.id}")
        pixel_region_table_reset_slots(grid.pixel_region_table, cpixel.x, cpixel.y)
        if debug:
            log.debug(
                f"pixel_region_table_insert_region: ({x}, {y}) "
                f"[{selected_region_ranked.region.id}] {selected_region_ranked.rank}"
            )
        pixel_region_table_insert_region(
            grid.pixel_region_table,
            x,
            y,
            selected_region_ranked.region,
            selected_region_ranked.rank,
        )
        # SR_TMP <
        if grid.pixel_status_table is not NULL:
            grid.pixel_status_table[x][y] = 1
        # grid.pixel_status_table[x][y] = 1
        # SR_TMP >

    if debug:
        log.debug("------------------------------------")

    # Cleanup
    free(selected_regions.slots)

    if debug:
        log.debug("> regiongrow_resolve_multi_assignments")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::regiongrow_resolve_multi_assignments
# :call: v --- calling ---
# :call: v stormtrack::core::identification::dbg_print_selected_regions
# :call: v stormtrack::core::identification::resolve_multi_assignment_best_connected_region
# :call: v stormtrack::core::identification::resolve_multi_assignment_biggest_region
# :call: v stormtrack::core::identification::resolve_multi_assignment_mean_strongest_region
# :call: v stormtrack::core::identification::resolve_multi_assignment_strongest_region
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegionRankSlot
# :call: v stormtrack::core::structs::cRegionRankSlots
cdef cRegionRankSlot* resolve_multi_assignment(
    cPixel* cpixel,
    cRegionRankSlots* selected_regions,
    int n_regions_max,
    cGrid* grid,
    int n_neighbors_max,
    bint debug,
) except? NULL:

    # ------------------------------------------------------------
    # (1) The region with most connections wins (direct > indirect)
    # ------------------------------------------------------------
    if debug:
        log.debug("(1) The region with most connections wins (direct > indirect)")
    resolve_multi_assignment_best_connected_region(
        cpixel, selected_regions, grid, n_neighbors_max,
    )
    if debug:
        dbg_print_selected_regions(1, selected_regions)
    if selected_regions.n == 1:
        return &selected_regions.slots[0]

    # ------------------------------------------------------------
    # (2) The biggest region wins (no. pixels)
    # ------------------------------------------------------------
    if debug:
        log.debug("(2) The biggest region wins (no. pixels)")
    resolve_multi_assignment_biggest_region(cpixel, selected_regions)
    if debug:
        dbg_print_selected_regions(2, selected_regions)
    if selected_regions.n == 1:
        return &selected_regions.slots[0]

    # ------------------------------------------------------------
    # (3) The strongest region wins (sum of pixel values)
    # ------------------------------------------------------------
    if debug:
        log.debug("(3) The strongest region wins (sum of pixel values)")
    resolve_multi_assignment_strongest_region(cpixel, selected_regions)
    if debug:
        dbg_print_selected_regions(2, selected_regions)
    if selected_regions.n == 1:
        return &selected_regions.slots[0]

    # ------------------------------------------------------------
    # (4) The 'mean strongest' region wins (mean of pixel values)
    # ------------------------------------------------------------
    if debug:
        log.debug("(4) The 'mean strongest' region wins (mean of pixel values)")
    resolve_multi_assignment_mean_strongest_region(cpixel, selected_regions)
    if debug:
        dbg_print_selected_regions(2, selected_regions)
    if selected_regions.n == 1:
        return &selected_regions.slots[0]

    # Pixel could not be assigned!
    raise Exception(
        f"pixel ({cpixel.x}, {cpixel.y}) could not be assigned unambiguously "
        "to a region\n\nregions"
    )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionRankSlots
# DBG_PERMANENT<<<
cdef void dbg_print_selected_regions(
    int i_test, cRegionRankSlots* selected_regions,
) except *:
    cdef cRegion* cregion
    cdef list foo
    if selected_regions.n == 1:
        cregion = selected_regions.slots[0].region
        log.info(f"[{i_test}] found best region: {cregion.id}")
    elif selected_regions.n > 1:
        foo = []
        for i in range(selected_regions.n):
            foo.append(selected_regions.slots[i].region.id)
        log.info(
            f"[{i_test}] still multiple regions: "
            f"{', '.join([str(i) for i in foo])}"
        )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: v --- calling ---
# :call: v stormtrack::core::identification::cpixel_count_neighbors_in_cregion
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegionRankSlots
# :call: v stormtrack::core::tables::cregion_rank_slots_insert_region
# :call: v stormtrack::core::tables::cregion_rank_slots_reset
cdef void resolve_multi_assignment_best_connected_region(
    cPixel* cpixel,
    cRegionRankSlots* selected_regions,
    cGrid* grid,
    int n_neighbors_max,
) except *:
    """Select the region(s) with most connections wins (direct > indirect).

    For every region, count the direct and indirect connections.
    The region with the most direct connections wins.
    Indirect connections form the tie-breaker.
    If both are tied, move on to the next criterion.

    """
    cdef bint debug = False
    if debug:
        log.debug(
            "< resolve_multi_assignment_best_connected_region "
            f"({cpixel.x}, {cpixel.y}"
        )
    cdef int i
    cdef int i_slot
    cdef int i_region
    cdef np.int32_t x = cpixel.x
    cdef np.int32_t y = cpixel.y
    cdef int n_direct_neighbors
    cdef int n_indirect_neighbors
    cdef int neighbor_rank
    cdef int n_direct_neighbors_max=0
    cdef int n_indirect_neighbors_max=0
    for i_slot in range(grid.pixel_region_table[x][y].max):
        cregion = grid.pixel_region_table[x][y].slots[i_slot].region
        if cregion is NULL:
            break

        # Count all neighbors; distinguish direct and indirect ones
        cpixel_count_neighbors_in_cregion(
            cpixel, cregion, &n_direct_neighbors, &n_indirect_neighbors, grid,
        )
        if n_direct_neighbors > 0:
            neighbor_rank = 0
        else:
            neighbor_rank = 1
        if debug:
            log.debug(
                f" [{cregion.id}] {n_direct_neighbors} direct, "
                f"{n_indirect_neighbors} indirect"
            )

        # Check if the region beats the previous leader
        if n_direct_neighbors < n_direct_neighbors_max or (
            n_direct_neighbors == n_direct_neighbors_max
            and n_indirect_neighbors < n_indirect_neighbors_max
        ):
            # Nope; discard this region
            continue
        elif (
            n_direct_neighbors > n_direct_neighbors_max
            or n_indirect_neighbors > n_indirect_neighbors_max
        ):
            # Yep; replace other region(s)
            cregion_rank_slots_reset(selected_regions)
            cregion_rank_slots_insert_region(selected_regions, cregion, neighbor_rank)
            n_direct_neighbors_max = n_direct_neighbors
            n_indirect_neighbors_max = n_indirect_neighbors
        else:
            # Tie; append this region
            cregion_rank_slots_insert_region(selected_regions, cregion, neighbor_rank)

    if debug:
        log.debug(
            f"> resolve_multi_assignment_best_connected_region: {selected_regions.n} "
            f"region{'' if selected_regions.n == 1 else 's'}"
        )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionRankSlots
# :call: v stormtrack::core::tables::cregion_rank_slots_copy
# :call: v stormtrack::core::tables::cregion_rank_slots_insert_region
# :call: v stormtrack::core::tables::cregion_rank_slots_reset
cdef void resolve_multi_assignment_biggest_region(
    cPixel* cpixel, cRegionRankSlots* selected_regions,
) except *:
    # print("> resolve_multi_assignment_biggest_region")
    cdef int i
    cdef int i_slot
    cdef int neighbor_rank
    cdef int size_max = -1
    cdef int n_selected_regions = 0
    cdef cRegion* cregion
    cdef cRegionRankSlots pre_selected_regions
    pre_selected_regions = cregion_rank_slots_copy(selected_regions)
    cregion_rank_slots_reset(selected_regions)
    for i_slot in range(selected_regions.max):
        cregion = pre_selected_regions.slots[i_slot].region
        neighbor_rank = pre_selected_regions.slots[i_slot].rank
        if cregion is NULL:
            continue
        # print(f" -> region {i_region}: {cregion.pixels_n}")
        if cregion.pixels_n < size_max:
            continue
        elif cregion.pixels_n == size_max:
            cregion_rank_slots_insert_region(selected_regions, cregion, neighbor_rank)
        else:
            cregion_rank_slots_reset(selected_regions)
            cregion_rank_slots_insert_region(selected_regions, cregion, neighbor_rank)
            size_max = cregion.pixels_n
    free(pre_selected_regions.slots)
    # print(
    #     f"< resolve_multi_assignment_biggest_region: {selected_regions.n} "
    #     f"region{'' if selected_regions.n == 1 else 's'}"
    # )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionRankSlots
# :call: v stormtrack::core::tables::cregion_rank_slots_copy
# :call: v stormtrack::core::tables::cregion_rank_slots_insert_region
# :call: v stormtrack::core::tables::cregion_rank_slots_reset
cdef void resolve_multi_assignment_strongest_region(
    cPixel* cpixel, cRegionRankSlots* selected_regions,
) except *:
    # print("> resolve_multi_assignment_strongest_region")
    cdef int i
    cdef int i_slot
    cdef int i_pixel
    cdef int n_selected_regions = 0
    cdef int neighbor_rank
    cdef cRegion* cregion
    cdef np.float32_t vsum
    cdef np.float32_t vsum_max = -1
    cdef cRegionRankSlots pre_selected_regions
    pre_selected_regions = cregion_rank_slots_copy(selected_regions)
    cregion_rank_slots_reset(selected_regions)
    for i_slot in range(selected_regions.max):
        cregion = pre_selected_regions.slots[i_slot].region
        neighbor_rank = pre_selected_regions.slots[i_slot].rank
        if cregion is NULL:
            continue
        vsum = 0.0
        for i_pixel in range(cregion.pixels_max):
            if cregion.pixels[i_pixel] is NULL:
                continue
            vsum += cregion.pixels[i_pixel].v
        # print(f" -> region {i_region}: {vsum}")
        if vsum < vsum_max:
            continue
        elif vsum == vsum_max:
            cregion_rank_slots_insert_region(selected_regions, cregion, neighbor_rank)
        else:
            cregion_rank_slots_reset(selected_regions)
            cregion_rank_slots_insert_region(selected_regions, cregion, neighbor_rank)
            vsum_max = vsum
    free(pre_selected_regions.slots)

    # print(
    #     f"< resolve_multi_assignment_strongest_region: {selected_regions.n} "
    #     f"region{'' if selected_regions.n == 1 else 's'}"
    # )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionRankSlots
# :call: v stormtrack::core::tables::cregion_rank_slots_copy
# :call: v stormtrack::core::tables::cregion_rank_slots_insert_region
# :call: v stormtrack::core::tables::cregion_rank_slots_reset
cdef void resolve_multi_assignment_mean_strongest_region(
    cPixel* cpixel, cRegionRankSlots* selected_regions,
) except *:
    # print("> resolve_multi_assignment_mean_strongest_region")
    cdef int i
    cdef int i_slot
    cdef int i_pixel
    cdef int n_selected_regions = 0
    cdef int neighbor_rank
    cdef cRegion* cregion
    cdef np.float32_t vmean
    cdef np.float32_t vmean_max = -1
    cdef cRegionRankSlots pre_selected_regions
    pre_selected_regions = cregion_rank_slots_copy(selected_regions)
    cregion_rank_slots_reset(selected_regions)
    for i_slot in range(selected_regions.max):
        cregion = pre_selected_regions.slots[i_slot].region
        neighbor_rank = pre_selected_regions.slots[i_slot].rank
        if cregion is NULL:
            continue
        vmean = 0.0
        for i_pixel in range(cregion.pixels_max):
            if cregion.pixels[i_pixel] is NULL:
                continue
            vmean += cregion.pixels[i_pixel].v
        with cython.cdivision(True):
            vmean /= (<np.float32_t>cregion.pixels_n)
        # print(f" -> region {i_region}: {vmean}")
        if vmean < vmean_max:
            continue
        elif vmean == vmean_max:
            vmean_max = vmean
            cregion_rank_slots_insert_region(selected_regions, cregion, neighbor_rank)
        else:
            cregion_rank_slots_reset(selected_regions)
            cregion_rank_slots_insert_region(selected_regions, cregion, neighbor_rank)
    free(pre_selected_regions.slots)

    # print(
    #     f"< resolve_multi_assignment_mean_strongest_region: {selected_regions.n} "
    #     f"region{'' if selected_regions.n == 1 else 's'}"
    # )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::resolve_multi_assignment_best_connected_region
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cpixel_get_neighbor
cdef void cpixel_count_neighbors_in_cregion(
    cPixel* cpixel,
    cRegion* cregion,
    int* n_direct_neighbors,
    int* n_indirect_neighbors,
    cGrid* grid,
) except *:
    cdef bint debug = False
    if debug:
        log.debug(
            f"< cpixel_count_neighbors_in_cregion: ({cpixel.x}, {cpixel.y})"
            f"[{'-' if cpixel.region is NULL else cpixel.region.id}] [{cregion.id}]"
        )
    cdef int i_neighbor
    cdef bint direct_neighbor
    cdef cPixel* neighbor
    cdef np.int8_t status
    # Loop over all neighbors and check whether they
    n_direct_neighbors[0] = 0
    n_indirect_neighbors[0] = 0
    for i_neighbor in range(grid.constants.n_neighbors_max):
        # always consider all 8 neighbors (8-connectivity)
        neighbor = cpixel_get_neighbor(
            cpixel,
            i_neighbor,
            grid.pixels,
            grid.constants.nx,
            grid.constants.ny,
            connectivity=8,
        )
        if neighbor is NULL:
            continue
        with cython.cdivision(True):
            if i_neighbor%2 == 0:
                direct_neighbor = True
            else:
                direct_neighbor = False
        if debug:
            log.debug(
                f"neighbor {i_neighbor}: ({neighbor.x}, {neighbor.y})"
                f"[{'-' if neighbor.region is NULL else neighbor.region.id}]"
            )
        if neighbor.region is NULL or neighbor.region.id != cregion.id:
            continue

        # Get status:
        #   -  0: assigned earlier to the region
        #   - -1: seed pixel of the region
        if grid.pixel_status_table is NULL:
            status = 0
        else:
            status = grid.pixel_status_table[neighbor.x][neighbor.y]
        if -1 <= status <= 0:
            if direct_neighbor:
                n_direct_neighbors[0] += 1
                if debug:
                    log.debug(f" => direct neighbor no. {n_direct_neighbors[0]}")
            else:
                n_indirect_neighbors[0] += 1
                if debug:
                    log.debug(" => indirect neighbor no. {n_direct_neighbors[0]}")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_insert_pixel
@cython.profile(False)
cdef bint regiongrow_assign_pixel(
    cPixel* cpixel,
    int neighbor_rank,
    cRegion* cregion,
    cRegion* pixels_multi_assigned,
    int n_seeds,
    cGrid* grid,
) except -1:
    """If the pixel is assignable, assign it to a region (or multiple)."""
    cdef int i_slot
    cdef int i
    cdef bint newly_assigned
    x = cpixel.x
    y = cpixel.y
    # print(f"< regiongrow_assign_pixel ({x}, {y})")

    # We're done if the pixel is not assignable:
    # - < 0: never been assignable (e.g. background or seed pixel)
    # - 0: definitively assigned in earlier iteration
    # - 1: definitively assigned in this iteration
    if grid.pixel_status_table[x][y] <= 1:
        return False

    # Pixel has already been provisionally assigned in this iteration
    elif grid.pixel_status_table[x][y] == 2:
        newly_assigned = False

    # Pixel has not yet been assigned
    elif grid.pixel_status_table[x][y] == 3:
        newly_assigned = True
        grid.pixel_status_table[x][y] = 2

    else:
        raise Exception(
            f"error: ({x}, {y}): invalid pixel status {grid.pixel_status_table[x][y]}"
        )

    # Get first empty slot to store region id
    # Also check if the pixel has already been assigned to this region
    for i_slot in range(grid.pixel_region_table[x][y].max):

        # Slot empty
        if grid.pixel_region_table[x][y].slots[i_slot].region is NULL:
            break

        # Pixel has already been assigned to this region
        if grid.pixel_region_table[x][y].slots[i_slot].region.id == cregion.id:
            newly_assigned = False
            if grid.pixel_region_table[x][y].slots[i_slot].rank == 0:
                return newly_assigned
            break

    grid.pixel_region_table[x][y].slots[i_slot].rank = neighbor_rank
    grid.pixel_region_table[x][y].slots[i_slot].region = cregion

    if i_slot > 0:

        # Add pixel to the multi-assigned list (unless it's alread there)
        for i in range(pixels_multi_assigned.pixels_max):
            if pixels_multi_assigned.pixels[i] is NULL:
                continue
            if pixels_multi_assigned.pixels[i].id == cpixel.id:
                break
        else:
            cregion_insert_pixel(
                pixels_multi_assigned, cpixel, link_region=False, unlink_pixel=False,
            )

    # print("> regiongrow_assign_pixel")
    return newly_assigned


# :call: > --- callers ---
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::c_find_extrema_2d
cpdef find_minima_2d(fld, n=4, nmax_extrema=100):
    if n not in [4, 8, 12, 20]:
        raise ValueError(f"Invalid parameter n={n}")
    return c_find_extrema_2d(fld, n, -1, nmax_extrema)


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::c_find_extrema_2d
cpdef find_maxima_2d(fld, n=4, nmax_extrema=100):
    if n not in [4, 8, 12, 20]:
        raise ValueError(f"Invalid parameter n={n}")
    return c_find_extrema_2d(fld, n, 1, nmax_extrema)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::find_maxima_2d
# :call: > stormtrack::core::identification::find_minima_2d
# :call: v --- calling ---
# :call: v stormtrack::core::identification::_c_find_extrema_2d_core
cdef np.ndarray[np.int32_t, ndim=2] c_find_extrema_2d(
    np.float32_t[:, :] fld, int n, np.float32_t sign, int nmax_extrema,
):
    cdef int i
    cdef int n_extrema
    cdef np.ndarray[np.int32_t, ndim=2] extrema = np.empty(
        [nmax_extrema, 2], dtype=np.int32,
    )
    n_extrema = _c_find_extrema_2d_core(fld, n, extrema, nmax_extrema, sign)
    return extrema[:n_extrema, :]


# :call: > --- callers ---
# :call: > stormtrack::core::identification::c_find_extrema_2d
# :call: v --- calling ---
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _c_find_extrema_2d_core(
    np.float32_t[:, :] fld,
    int n,
    np.int32_t  [:, :] extrema,
    int max_extrema,
    np.float32_t sign,
) except -1:
    cdef int i
    cdef int j
    cdef int nxs
    cdef int nxe
    cdef int nys
    cdef int nye
    cdef int n_extrema = 0
    cdef np.int32_t nx = fld.shape[0]
    cdef np.int32_t ny = fld.shape[1]
    cdef np.float32_t v
    if n == 4 or n == 8:
        nxs = 1
        nys = 1
        nxe = nx - 1
        nye = ny - 1
    elif n == 12 or n == 20:
        nxs = 2
        nys = 2
        nxe = nx - 2
        nye = ny - 2
    for i in range(nxs, nxe):
        for j in range(nys, nye):
            v = fld[i, j]
            if n >= 4:
                if (
                    sign*v <= sign*fld[i-1, j  ]
                    or sign*v <= sign*fld[i  , j+1]
                    or sign*v <= sign*fld[i+1, j  ]
                    or sign*v <= sign*fld[i  , j-1]
                ):
                    continue
            if n >= 8:
                if (
                    sign*v <= sign*fld[i-1, j+1]
                    or sign*v <= sign*fld[i+1, j+1]
                    or sign*v <= sign*fld[i+1, j-1]
                    or sign*v <= sign*fld[i-1, j-1]
                ):
                    continue
            if n >= 12:
                if (
                    sign*v <= sign*fld[i-2, j  ]
                    or sign*v <= sign*fld[i  , j+2]
                    or sign*v <= sign*fld[i+2, j  ]
                    or sign*v <= sign*fld[i  , j-2]
                ):
                    continue
            if n >= 20:
                if (
                    sign*v <= sign*fld[i-2, j-1]
                    or sign*v <= sign*fld[i-2, j+1]
                    or sign*v <= sign*fld[i-1, j+2]
                    or sign*v <= sign*fld[i+1, j+2]
                    or sign*v <= sign*fld[i+2, j+1]
                    or sign*v <= sign*fld[i+2, j-1]
                    or v >= fld[i+1, j-2]
                    or v >= fld[i-1, j-2]
                ):
                    continue
            extrema[n_extrema][0] = i
            extrema[n_extrema][1] = j
            n_extrema += 1
        if n_extrema == max_extrema:
            raise Exception("maximum number of extrema reached")
    return n_extrema


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::tracking::FeatureTracker::extend_tracks
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::identification::dbg_check_features_cregion_pixels
# :call: v stormtrack::core::identification::dbg_features_check_unique_pixels
# :call: v stormtrack::core::identification::feature_to_cregion
# :call: v stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::grid::grid_create_cregion
cdef void features_to_cregions(
    list features,
    int n_regions,
    cRegions* cregions,
    cRegionConf cregion_conf,
    bint ignore_missing_neighbors,
    cGrid* grid,
    cConstants* constants,
    bint reuse_feature_cregion=False,
) except *:
    """Turn a list of Feature objects into a cRegions object"""
    cdef str _name_ = "features_to_cregions"
    cdef bint debug = False
    if debug:
        log.debug(f"< {_name_}: {len(features)} features")
        log.debug(f"{pformat(features)}")
    cdef int i
    cdef int j
    cdef Feature feature
    cdef cRegion* cregion
    cdef bint determine_boundaries_individually = False
    cdef np.uint64_t fid
    dbg_features_check_unique_pixels(features)

    # Create cRegion objects
    cdef np.uint64_t* fids = <np.uint64_t*>malloc(n_regions*sizeof(np.uint64_t))
    for i in range(n_regions):
        feature = features[i]
        if debug:
            log.debug(
                f"convert feature {i + 1}/{n_regions}: "
                f"[{n_regions}/]({feature.id}/{feature.n})"
            )

        # Check uniqueness of feature id
        fid = feature.id
        for j in range(i):
            if fids[j] == fid:
                raise Exception(f"# {i}: duplicate feature id: {fid} (# {j})")
        fids[i] = fid

        # Initialize new cregion (and check that there is not already one)
        if feature.cregion is not NULL:
            msg = f"warning: {_name_}: feature {feature.id}: cregion not NULL: "
            if reuse_feature_cregion:
                log.warning(msg+"reuse existing")
                continue
            else:
                log.warning(f"{msg} clean up existing")
                feature.cleanup_cregion(unlink_pixels=False, reset_connected=False)
        cregion = grid_create_cregion(grid)
        # if debug: log.debug("create new cregion at {}".format(<long>cregion))
        cregions_link_region(cregions, cregion, cleanup=False, unlink_pixels=False)

        # Create cregion from feature
        feature_to_cregion(
            feature,
            cregion,
            cregion_conf,
            grid,
            determine_boundaries_individually,
            constants,
        )
        if debug:
            log.debug(
                f" -> feature({feature.id}) to cregion({feature.cregion.id}) "
                f"({feature.cregion.pixels_n} pixels)"
            )
    free(fids)

    # Add neighbors (i.e. connected regions)
    if debug:
        log.debug("add neighbors (i.e. connected regions)")
    features_neighbors_to_cregions_connected(
        features, cregions, ignore_missing_neighbors,
    )

    # Determine all boundaries at once
    if not determine_boundaries_individually:
        if debug:
            log.debug("determine boundaries of all cfeatures")
        cregions_determine_boundaries(cregions, grid)

    dbg_check_features_cregion_pixels(features) # SR_DBG_PERMANENT


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_to_cregions
# :call: v --- calling ---
# SR_DBG <<<
def dbg_features_check_unique_pixels(features):
    # Check uniqueness of pixels
    pixels_all = [(x, y) for f in features for x, y in f.pixels]
    if len(set(pixels_all)) < len(pixels_all):
        shared_fids = {}
        for feature1 in features:
            pxs1 = set([(x, y) for x, y in feature1.pixels])
            for feature2 in features:
                if feature1 == feature2 or (feature2.id, feature1.id) in shared_fids:
                    continue
                pxs2 = set([(x, y) for x, y in feature2.pixels])
                shared = sorted(pxs1.intersection(pxs2))
                if len(shared) > 0:
                    shared_fids[(feature1.id, feature2.id)] = shared
        err = f"{len(shared_fids)} feature pairs share pixels:\n" + "\n".join([
                f" [{fid1}]<->[{fid2}]: {', '.join([f'({x}, {y})' for x, y in pxs])}"
                for (fid1, fid2), pxs in sorted(shared_fids.items())
            ])
        print("warning: "+err)
        # raise ValueError(err)
        outfile = "cyclone_tracking_debug_shared_pixels.txt"
        print("debug: write feature ids and pixels to {}".format(outfile))
        with open(outfile, "a") as fo:
            fo.write(err+"\n")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_to_cregions
# :call: v --- calling ---
# SR_DBG <<<
cpdef void dbg_check_features_cregion_pixels(list features) except *:
    cdef Feature feature
    for feature in features:
        if feature.cregion is not NULL:
            if feature.n != feature.cregion.pixels_n:
                err = (
                    f"feature [{feature.id}/{feature.cregion.id}]: inconsistent "
                    f"pixels: ({feature.n}/{feature.cregion.pixels_n})"
                )
                print("warning: "+err)
                # raise Exception(err)
                outfile = "cyclone_tracking_debug_features_inconsistent_pixels.txt"
                print(f"debug: write feature info to {outfile}")
                with open(outfile, "a") as fo:
                    fo.write(err+"\n")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_to_cregions
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_connect_with
cdef void features_neighbors_to_cregions_connected(
    list features_list, cRegions* cregions, bint ignore_missing_neighbors,
) except *:
    cdef bint debug = False
    cdef str _name_ = "features_neihbors_to_cregions_connected"
    cdef int i_region
    cdef int n_regions=len(features_list)
    cdef int n_neighbors
    cdef Feature feature
    cdef Feature feature_neighbor
    cdef Feature feature_neighbor_2
    cdef cRegion* cregion
    cdef cRegion* cregion_neighbor
    if not n_regions == cregions.n:
        raise Exception(
            f"{_name_}: numbers of features ({n_regions}) and cregions "
            f"({cregions.n}) differ"
        )
    cdef int i_region_2
    for i_region in range(cregions.n):
        feature = features_list[i_region]
        n_neighbors  = len(feature.neighbors)
        if debug:
            log.debug(f"feature {feature.id}: {n_neighbors} neighbors")
        if n_neighbors == 0:
            continue
        cregion = cregions.regions[i_region]
        for i_neighbor in range(n_neighbors):
            feature_neighbor = feature.neighbors[i_neighbor]
            # Sanity check
            if feature_neighbor.id == feature.id:
                raise Exception(
                    f"feature {feature.id} cannot be it's own neighbor!"
                )

            # SR_TODO use table to cross-reference features and cregions
            # SR_TODO (as is already done to transfer connected features
            # SR_TODO to neighbors, i.e. just the opposite direction)

            # Find array index of neighbor
            for i_region_2 in range(cregions.n):
                if i_region_2 == i_region:
                    continue
                feature_neighbor_2 = features_list[i_region_2]
                if feature_neighbor_2.id == feature_neighbor.id:
                    cregion_neighbor = cregions.regions[i_region_2]
                    break
            else:
                if not ignore_missing_neighbors:
                    raise Exception(
                        f"feature {feature.id}: missing neighbor {feature_neighbor.id}"
                    )
                continue
            cregion_connect_with(cregion, cregion_neighbor)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_to_cregions
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: v stormtrack::core::cregion::cregion_init
# :call: v stormtrack::core::cregion::cregion_insert_pixel
cdef void feature_to_cregion(
    Feature feature,
    cRegion* cregion,
    cRegionConf cregion_conf,
    cGrid* grid,
    bint determine_boundaries,
    cConstants* constants,
) except *:
    """Turn a Feature object into cRegion object.

    Determining the feature boundaries is optional because it is expensive
    for single cRegions. In case of multiple conversions, it is much more
    efficient to determine the boundaries in bulk after the conversion to
    cRegions.

    """
    cdef bint debug = False
    cdef str _name_ = "feature_to_cregion"
    if debug:
        log.debug(f"< feature_to_cregion ({feature.id})")
    cdef int i
    cdef int n_pixels = feature.n
    cdef np.int32_t x
    cdef np.int32_t y
    cdef bint has_values = feature.values.size > 0
    # DBG_BLOCK <
    if debug:
        log.debug(f"  {n_pixels} pixels:")
        for i, (x, y) in enumerate(feature.pixels):
            log.debug(f"  {i + 1}/{n_pixels}\t({x}, {y})")
    # DBG_BLOCK >
    cregion_init(cregion, cregion_conf, feature.id)
    for i in range(n_pixels):
        x = feature.pixels[i, 0]
        y = feature.pixels[i, 1]

        if x < 0 or y < 0:
            raise Exception(f"{_name_}: invalid pixel # {i}: ({x}, {y})")

        if x >= grid.constants.nx or y >= grid.constants.ny:
            raise Exception(
                f"{_name_}: pixel # {i} outside "
                f"{grid.constants.nx}x{grid.constants.ny} grid: ({x}, {y})"
            )

        if has_values:
            grid.pixels[x][y].v = feature.values[i]
        else:
            grid.pixels[x][y].v = -1

        # Note: unlink_pixel prevents 'shell pixel connected to wrong region'
        cregion_insert_pixel(
            cregion, &grid.pixels[x][y], link_region=True, unlink_pixel=True,
        )

    if determine_boundaries:
        cregion_determine_boundaries(cregion, grid)

    # Link cregion to feature
    feature.set_cregion(cregion)

    # SR_TMP <
    # Check that the number of pixels match
    if feature.cregion is not NULL:
        if feature.n != feature.cregion.pixels_n:
            err = (
                f"feature {feature.id}: inconsistent pixels: "
                f"{feature.n}/{feature.cregion.pixels_n}"
            )
            print("warning: "+err)
            # raise Exception(err)
            outfile = "cyclone_tracking_debug_cfeature_inconsistent_pixels.txt"
            print(f"debug: write feature info to {outfile}")
            with open(outfile, "a") as fo:
                fo.write(err+"\n")
    # SR_TMP >

    if debug:
        log.debug(f"> feature_to_cregion ({feature.id})")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::Feature::derive_boundaries_from_pixels
# :call: > stormtrack::core::identification::Feature::derive_holes_from_pixels
# :call: > stormtrack::core::identification::Feature::derive_shells_from_pixels
# :call: > test_stormtrack::test_core::test_features::test_boundaries::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
# :call: v stormtrack::core::cregion::cregion_insert_pixels_coords
# :call: v stormtrack::core::constants::default_constants
# :call: v stormtrack::core::grid::grid_cleanup
# :call: v stormtrack::core::grid::grid_create
cpdef tuple pixels_find_boundaries(
    np.ndarray[np.int32_t, ndim=2] pixels,
    Constants constants = None,
    np.int32_t nx = 0,
    np.int32_t ny = 0,
):
    """Find outer and inner boundaries of features from the raw coordinates.

    This python-accessible function has been written initially in order to
    unit test the core function.

    """
    # print("< pixels_find_boundaries")

    cdef int i
    cdef int j

    # Set up constants
    if constants is None:
        if nx == 0 or ny == 0:
            raise Exception("must pass either (nx, ny) or constants")
        constants = default_constants(nx=nx, ny=ny)
    else:
        nx = constants.nx
        ny = constants.ny
    cdef cConstants* cconstants = constants.to_c()

    # Initialize grid
    cdef np.ndarray[np.float32_t, ndim=2] field = np.zeros(
        [cconstants.nx, cconstants.ny], dtype=np.float32,
    )
    cdef cGrid grid = grid_create(field, cconstants[0])

    # Initialize feature pixels
    cdef cRegion cregion
    cdef cRegionConf cregion_conf = cRegionConf(
        connected_max=0,
        pixels_max=pixels.shape[0],
        shells_max=1,
        shell_max=pixels.shape[0],
        hole_max=0,
        holes_max=0,
    )
    cregion_init(&cregion, cregion_conf, cregion_get_unique_id())
    cregion.id = 1
    cregion_insert_pixels_coords(
        &cregion, grid.pixels, pixels, link_region=True, unlink_pixels=False,
    )

    # Allocate neighbor link status table
    neighbor_link_stat_table_alloc(&grid.neighbor_link_stat_table, cconstants)

    # Determine boundary pixels
    cregion_determine_boundaries(&cregion, &grid)

    # Extract shell
    cdef list shells = []
    cdef np.ndarray[np.int32_t, ndim=2] shell
    for i in range(cregion.shells_n):
        shell = np.empty([cregion.shell_n[i], 2], dtype=np.int32)
        for j in range(cregion.shell_n[i]):
            shell[j, 0] = cregion.shells[i][j].x
            shell[j, 1] = cregion.shells[i][j].y
        shells.append(shell.copy())

    # Extract holes
    cdef list holes = []
    cdef np.ndarray[np.int32_t, ndim=2] hole
    for i in range(cregion.holes_n):
        hole = np.empty([cregion.hole_n[i], 2], dtype=np.int32)
        for j in range(cregion.hole_n[i]):
            hole[j, 0] = cregion.holes[i][j].x
            hole[j, 1] = cregion.holes[i][j].y
        holes.append(hole.copy())

    grid_cleanup(&grid)

    return shells, holes


# :call: > --- callers ---
# :call: > stormtrack::core::identification::Field2D::__init__
# :call: > stormtrack::core::identification::Field2D::get
# :call: > stormtrack::core::identification::Field2D::list
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Field2D
cdef class Pixel:

    def __init__(Pixel self, int x, int y, float v, Field2D fld, np.uint64_t id):
        self.id = id
        self.x = x
        self.y = y
        self.v = v
        self.fld = fld
        self.type = pixeltype_none
        self.region = -1

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.v}, {self.type})"

    cpdef list neighbors(Pixel self):
        if not self.fld:
            return []
        return self.fld.get_neighbors(self)


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Pixel
cdef class Field2D:

    def __init__(Field2D self, fld_raw):
        fld_raw = np.asarray(fld_raw)
        self.nx, self.ny = fld_raw.shape
        self._fld = [[None]*self.ny for _ in range(self.nx)]
        cdef int pid = 0
        for (i, j), z in np.ndenumerate(fld_raw):
            self._fld[i][j] = Pixel(i, j, z, fld=self, id=pid)
            pid += 1

    def __iter__(self):
        return iter([p for pp in self._fld for p in pp])

    cdef Pixel get(Field2D self, int i, int j):
        return self._fld[i][j]

    cdef list get_neighbors(Field2D self, Pixel pixel):
        cdef int j = pixel.y
        cdef int i = pixel.x
        cdef list neighbors = []
        if i < self.nx-1:
            neighbors.append(self.get(i + 1, j))
        if j > 0:
            neighbors.append(self.get(i, j - 1))
        if i > 0:
            neighbors.append(self.get(i - 1, j))
        if j < self.ny-1:
            neighbors.append(self.get(i, j + 1))
        return neighbors

    cpdef list pixels(Field2D self):
        return list(iter(self))


_TRACK_REGISTER = {}


# :call: > --- callers ---
# :call: > stormtrack::core::identification::Feature::__reduce__
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
cpdef Feature Feature_rebuild(
    np.ndarray values,  # 0
    np.ndarray pixels,  # 1
    np.ndarray center,  # 2
    np.ndarray extrema,  # 3
    list shells,  # 4
    list holes,  # 5
    np.uint64_t id,  # 6
    np.uint64_t track_id,  # 7
    str vertex_name,  # 8
    np.uint64_t timestep,  # 9
    dict properties,  # 10
    dict associates,  # 11
    list neighbors,  # 12
    dict _cache,  # 13
    dict _shared_boundary_pixels,
    dict _shared_boundary_pixels_unique
):

    # Initialize feature
    feature = Feature(
        values=values,
        pixels=pixels,
        center=center,
        extrema=extrema,
        shells=shells,
        holes=holes,
        id_=id,
        _cache=_cache,
    )

    # Link track and vertex
    track = _TRACK_REGISTER.get(track_id)
    if track is not None:
        feature.set_track(track)
        if vertex_name is not None:
            feature._vertex_name = vertex_name

    # Set public attributes
    feature.timestep = timestep
    feature.properties = properties
    feature.associates = associates
    feature.neighbors = neighbors
    feature._shared_boundary_pixels = _shared_boundary_pixels
    feature._shared_boundary_pixels_unique = _shared_boundary_pixels_unique

    return feature


# :call: > --- callers ---
# :call: > stormtrack::core::identification::Feature_rebuild
# :call: > stormtrack::core::identification::_replace_feature_associations
# :call: > stormtrack::core::identification::associate_features
# :call: > stormtrack::core::identification::create_feature
# :call: > stormtrack::core::identification::cregion_find_corresponding_feature
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::cregions_create_features
# :call: > stormtrack::core::identification::cyclones_to_features
# :call: > stormtrack::core::identification::feature2d_from_jdat
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::features_associates_obj2id
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::features_neighbors_id2obj
# :call: > stormtrack::core::identification::features_neighbors_obj2id
# :call: > stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: > stormtrack::core::identification::features_reset_cregion
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::resolve_indirect_associations
# :call: > stormtrack::core::io::rebuild_features_core
# :call: > stormtrack::core::tracking::FeatureTrack::features_ts_ns
# :call: > stormtrack::core::tracking::FeatureTracker::_assign_successors
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
# :call: > stormtrack::core::tracking::FeatureTracker::_finish_track
# :call: > stormtrack::core::tracking::FeatureTracker::_start_track
# :call: > stormtrack::core::tracking::FeatureTracker::_swap_grids
# :call: > stormtrack::core::tracking::FeatureTracker::extend_tracks
# :call: > stormtrack::core::tracking::TrackFeatureMerger::_collect_merge_es_attrs_core
# :call: > stormtrack::core::tracking::TrackFeatureMerger::collect_merge_es_attrs
# :call: > stormtrack::core::tracking::TrackFeatureMerger::collect_merge_vs_attrs
# :call: > stormtrack::core::tracking::TrackFeatureMerger::collect_neighbors
# :call: > stormtrack::core::tracking::TrackFeatureMerger::collect_vertices_edges
# :call: > stormtrack::core::tracking::TrackFeatureMerger::merge_feature
# :call: > stormtrack::core::tracking::TrackFeatureMerger::replace_vertices_edges
# :call: > stormtrack::core::tracking::TrackFeatureMerger::run
# :call: > stormtrack::core::tracking::TrackableFeatureCombination_Oldstyle::overlaps
# :call: > stormtrack::core::tracking::TrackableFeature_Oldstyle
# :call: > stormtrack::core::tracking::dbg_check_features_cregion_pixels
# :call: > stormtrack::extra::front_surgery::*
# :call: > test_stormtrack::test_core::test_features::test_area_lonlat::*
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: > test_stormtrack::test_core::test_features::test_split_regiongrow::*
# :call: > test_stormtrack::utils::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature_rebuild
# :call: v stormtrack::core::identification::_feature__from_jdat__pixels_from_tables
# :call: v stormtrack::core::identification::pixels_find_boundaries
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::constants::default_constants
# :call: v stormtrack::core::utilities::NAN_UI64
cdef class Feature:

    def __cinit__(self,
        np.ndarray[np.int32_t, ndim=2] pixels = None,
        *,
        np.ndarray[np.float32_t, ndim=1] values = None,
        np.ndarray[np.int32_t, ndim=1] center = None,
        np.ndarray[np.int32_t, ndim=2] extrema = None,
        list shells = None,
        list holes = None,
        np.uint64_t id_ = 0,
        np.uint64_t timestep = NAN_UI64,
        bint derive_center = True,
        bint derive_boundaries = True,
        dict _cache = None,
    ):
        self.debug = False

        if pixels is not None and values is not None and values.size > 0:
            if pixels.size != 2*values.size:
                err = "Inconsistent dimensions: {}, {}".format(
                        np.asarray(pixels).shape, np.asarray(values).shape)
                raise ValueError(err)

        self.set_id(id_)

        self.set_pixels(pixels)
        self.set_values(values)

        # If no center is passed, it is derived from the pixels or shells
        self.set_center(center, derive=derive_center)
        self.set_extrema(extrema)

        # If no shell/holes is passed, it is derived from the pixels
        self.set_shells(shells, derive=derive_boundaries)
        self.set_holes(holes, derive=derive_boundaries)

        self._track_id = NAN_UI64
        self._vertex_name = None

        # Public properties
        self.timestep = timestep
        self.properties = {}
        self.associates = {}
        self._stats_lst = []
        self.neighbors = []
        self._shared_boundary_pixels = {}
        self._shared_boundary_pixels_unique = {}

        self._cache = {} if _cache is None else _cache

        self.set_cregion(NULL)

    def __reduce__(self):
        return Feature_rebuild, (
            self.values,  # 0
            self.pixels,  # 1
            self.center,  # 2
            self.extrema,  # 3
            self.shells,  # 4
            self.holes,  # 5
            self.id,  # 6
            self._track_id,  # 7
            self._vertex_name,  # 8
            self.timestep,  # 9
            self.properties,  # 10
            self.associates,  # 11
            self.neighbors,  # 12
            self._cache,  # 13
            self._shared_boundary_pixels,
            self._shared_boundary_pixels_unique,
        )

    def __dealloc__(self):
        # Note: log.debug cannot be used inside from __dealloc__
        if self.debug: print("{}.__dealloc__()".format(self))
        # print("WARNING: Feature.__dealloc__: cregion not cleaned up (see comment)")
        #
        # Cleanup of cregion here can interfere with cleanup of the same
        # cregion elsewhere (in pure cython code), causing segfault!
        # (At least that's my best guess...)
        #
        # TODO: Move all cregions to grod such that they are properly
        # centralized, and keep track of links between cregions and
        # Feature objects, such that all cregions are only cleaned up
        # exactly once!
        #
        # (Of course even better would be to slowly, but surely merge
        # cRegion and Feature, starting by elimination Pixel etc.)
        #
        # self.cleanup_cregion()

    def __repr__(self):
        s = "Feature["
        # s += f"{id(self)}|"
        s += f"{self.id}]("
        if self.timestep != NAN_UI64:
            s += f"@{self.timestep}, "
        s += f"n={self.n}/"
        if self.cregion is not NULL:
            s += f"{self.cregion.pixels_n}"
        # s += f", min={self.min:.3f}, mean={self.mean:.3f}, max={self.max.3f}"
        s += ")"
        return s

    def __hash__(self):
        return id(self)

    def __richcmp__(self, Feature other, int op):
        # <   0
        # <=  1
        # ==  2
        # !=  3
        # :call: >   4
        # >=  5
        if op == Py_LT:
            return self.id < other.id
        elif op == Py_EQ:
            return self.id == other.id
        elif op == Py_NE:
            return self.id != other.id
        elif op == Py_GT:
            return self.id > other.id
        else:
            err = "Feature comparison operator {}".format(op)
            raise NotImplementedError(err)

    def __iter__(self):
        return iter(self.pixels)

    def __len__(self):
        return len(self.pixels)

    def __copy__(self):
        feature = Feature(
            id_=self.id,
            timestep=self.timestep,
            values=copy(self.values),
            pixels=copy(self.pixels),
            center=copy(self.center),
            extrema=copy(self.extrema),
            shells=[copy(a) for a in self.shells],
            holes=[copy(a) for a in self.holes],
            _cache=copy(self._cache),
        )

        feature.properties = copy(self.properties)
        feature.associates = copy(self.associates)
        feature.neighbors  = copy(self.neighbors)
        feature._stats_lst = copy(self._stats_lst)

        feature._vertex_name = self._vertex_name
        feature._track_id = self._track_id
        feature._shared_boundary_pixels = copy(self._shared_boundary_pixels)
        feature._shared_boundary_pixels_unique = copy(self._shared_boundary_pixels_unique)

        assert feature == self  # SR_TMP

        return feature

    def __deepcopy__(self, memo):
        feature = copy(self)

        feature.properties = deepcopy(feature.properties, memo)

        # Copy associates
        old_associates = feature.associates
        feature.associates = {}
        for name, associates in old_associates.items():
            feature.associates[name] = []
            for associate in associates:
                if associate == self:
                    feature.associates[name].append(feature)
                else:
                    feature.associates[name].append(copy(associate))

        # Copy neighbors
        old_neighbors = feature.neighbors
        feature.neighbors = []
        for neighbor in old_neighbors:
            if neighbor == self:
                feature.neighbors.append(feature)
            else:
                feature.neighbors.append(copy(neighbor))

        return feature

    @property
    def stats(self):
        if self._stats_lst is None:
            self.compute_stats()
        return dict(self._stats_lst)

    def compute_stats(self):
        self._stats_lst = [("n", self.n)]
        if self.values is None or self.values.size == 0:
            self._stats_lst.extend(
                [
                    ("min", -1.0),
                    ("max", -1.0),
                    ("mean", -1.0),
                    ("median", -1.0),
                    ("sum", -1.0),
                    ("absmin", -1.0),
                    ("absmax", -1.0),
                    ("absmean", -1.0),
                    ("absmedian", -1.0),
                    ("abssum", -1.0),
                ]
            )
        else:
            self._stats_lst.extend(
                [
                    ("min", np.nanmin(self.values)),
                    ("max", np.nanmax(self.values)),
                    ("mean", np.nanmean(self.values)),
                    ("median", np.nanmedian(self.values)),
                    ("sum", np.nansum(self.values)),
                    ("absmin", np.nanmin(np.abs(self.values))),
                    ("absmax", np.nanmax(np.abs(self.values))),
                    ("absmean", np.nanmean(np.abs(self.values))),
                    ("absmedian", np.nanmedian(np.abs(self.values))),
                    ("abssum", np.nansum(np.abs(self.values))),
                ]
            )

    cpdef void set_id(self, np.uint64_t id_) except *:
        self.id = id_

    cpdef void set_pixels(self, np.ndarray[np.int32_t, ndim=2] pixels) except *:
        if pixels is None:
            self.pixels = np.array([[]], np.int32)
        else:
            self.pixels = pixels
            self.n = pixels.size/2
            if not self.center_is_valid():
                center = pixels.mean(axis=0).round().astype(np.int32)
                self.set_center(center)
        self.compute_stats()

    cpdef void set_values(self, np.ndarray[np.float32_t, ndim=1] values) except *:
        if values is None:
            self.values = np.array([], np.float32)
        else:
            self.values = values
        self.compute_stats()

    cpdef void set_shells(self, list shells, bint derive=True,
            bint derive_pixels=False, bint derive_center=False,
            int nx=-1, int ny=-1) except *:

        if shells is None:
            if derive:
                self.derive_shells_from_pixels()
            else:
                self.shells = []

        else:
            self.shells = shells

            if derive_pixels:
                self.derive_pixels_from_boundaries(nx, ny)

            if derive_center:
                # Note: Pixels must have been set previously, or derived above
                self.derive_center_from_pixels()

    cpdef void set_holes(self, list holes, bint derive=True) except *:
        if holes is None:
            if derive:
                if self.pixels is None:
                    err = "cannot derive holes without pixels (None)"
                    raise Exception(err)
                if self.pixels.size == 0:
                    err = "cannot derive holes without pixels (size == 0)"
                    raise Exception(err)
                self.derive_holes_from_pixels()
            else:
                self.holes = []
        else:
            self.holes = holes

    cpdef void set_center(
        self, np.ndarray[np.int32_t, ndim=1] center, bint derive=True,
    ) except *:
        if center is None:
            if derive:
                if self.pixels is None:
                    err = "cannot derive center without pixels (None)"
                    raise Exception(err)
                if self.pixels.size == 0:
                    err = "cannot derive center without pixels (size == 0)"
                    raise Exception(err)
                self.center = np.round(self.pixels.mean(axis=0)).astype(np.int32)
            else:
                self.center = np.array([-1, -1], np.int32)
        else:
            self.center = center

    def center_is_valid(self):
        if self.center is None:
            return False
        if self.center[0] < 0 or self.center[1] < 0:
            return False
        return True

    cpdef void set_extrema(self, np.ndarray[np.int32_t, ndim=2] extrema) except *:
        if extrema is None:
            self.extrema = np.array([[]], np.int32)
        else:
            self.extrema = extrema

    cpdef np.float32_t sum(self, bint abs=False) except -999:
        if self.values is None or self.values.size == 0:
            if abs:
                return self.stats["abssum"]
            else:
                return self.stats["sum"]
        values = self.values
        if abs:
            values = np.abs(values)
        return np.nansum(values)

    cpdef np.float32_t min(self, bint abs=False) except -999:
        if self.values is None or self.values.size == 0:
            if abs:
                return self.stats["absmin"]
            else:
                return self.stats["min"]
        values = self.values
        if abs:
            values = np.abs(values)
        return np.nanmin(values)

    cpdef np.float32_t mean(self, bint abs=False) except -999:
        if self.values is None or self.values.size == 0:
            if abs:
                return self.stats["absmean"]
            else:
                return self.stats["mean"]
        values = self.values
        if abs:
            values = np.abs(values)
        return np.nanmean(self.values)

    cpdef np.float32_t median(self, bint abs=False) except -999:
        if self.values is None or self.values.size == 0:
            if abs:
                return self.stats["absmedian"]
            else:
                return self.stats["median"]
        values = self.values
        if abs:
            values = np.abs(values)
        return np.nanmedian(self.values)

    cpdef np.float32_t max(self, bint abs=False) except -999:
        if self.values is None or self.values.size == 0:
            if abs:
                return self.stats["absmax"]
            else:
                return self.stats["max"]
        values = self.values
        if abs:
            values = np.abs(values)
        return np.nanmax(self.values)

    def replace_values(self, fld):
        px, py = self.pixels.T
        self.values = fld[px, py]

    def mirror_values(self):
        self.values = -self.values

    cdef void set_cregion(self, cRegion* cregion) except *:
        if self.debug:
            log.debug(
                f"{self}.set_cregion({'NULL' if cregion is NULL else cregion.id})"
            )
        self.cregion = cregion

    cpdef void reset_cregion(self, bint warn=True) except *:
        if self.debug: print("{}.reset_cregion({})".format(self, warn))
        if warn:
            log.warning(
                "Feature2d.reset_cregion: might cause memory leaks "
                "(use Feature.cleanup_cregion)"
            )
        self.set_cregion(NULL)

    cpdef void cleanup_cregion(self, unlink_pixels=True, reset_connected=True) except *:
        # Note: log.debug cannot be used if called from __dealloc__
        if self.debug:
            log.debug(
                f"{self}.cleanup_cregion"
                f"({'NULL' if self.cregion is NULL else self.cregion.id})"
            )
        if self.cregion is not NULL:
            cregion_cleanup(self.cregion, unlink_pixels, reset_connected)
            self.set_cregion(NULL)

    cpdef object track(self):
        return _TRACK_REGISTER.get(self._track_id)

    cpdef void set_track(self, object track) except *:
        _TRACK_REGISTER[track.id] = track
        self._track_id = track.id

    cpdef void unset_track(self) except *:
        self._track_id = NAN_UI64

    cpdef void hardcode_n_pixels(self, int n) except *:
        """Hard-code the number of pixels (if pixels not read from disk)."""
        self.n = n

    def _check_pixels_present(self, action):
        if self.pixels is None:
            err = f"cannot {action} without pixels (None)"
            raise Exception(err)
        if self.pixels.size == 0:
            err = f"cannot {action} without pixels (size == 0)"
            raise Exception(err)

    def derive_center_from_pixels(self):
        self._check_pixels_present("derive center")
        raise NotImplementedError(f"{type(self).__name__}.derive_center_from_pixels()")

    def derive_boundaries_from_pixels(self, constants=None):
        self._check_pixels_present("derive boundaries")
        if constants is None:
            nx, ny = self.pixels[:, 0].max() + 1, self.pixels[:, 1].max() + 1
            constants = default_constants(nx=nx, ny=ny)
        x0, y0, x1, y1 = self.bbox()
        ll = np.array([x0, y0])
        pixels_rel = self.pixels - ll
        shells_rel, holes_rel = pixels_find_boundaries(pixels_rel, constants)
        self.shells = [shell_rel + ll for shell_rel in shells_rel]
        self.holes = [hole_rel + ll for hole_rel in holes_rel]

    def derive_shells_from_pixels(self, constants=None):
        self._check_pixels_present("derive shells")
        if constants is None:
            nx, ny = self.pixels[:, 0].max() + 1, self.pixels[:, 1].max() + 1
            constants = default_constants(nx=nx, ny=ny)
        x0, y0, x1, y1 = self.bbox()
        ll = np.array([x0, y0])
        pixels_rel = self.pixels - ll
        shells_rel, holes_rel = pixels_find_boundaries(pixels_rel, constants)
        self.shells = [shell_rel + ll for shell_rel in shells_rel]

    def derive_holes_from_pixels(self, constants=None):
        self._check_pixels_present("derive holes")
        if constants is None:
            nx, ny = self.pixels[:, 0].max() + 1, self.pixels[:, 1].max() + 1
            constants = default_constants(nx=nx, ny=ny)
        x0, y0, x1, y1 = self.bbox()
        ll = np.array([x0, y0])
        pixels_rel = self.pixels - ll
        shells_rel, holes_rel = pixels_find_boundaries(pixels_rel, constants)
        self.holes = [hole_rel + ll for hole_rel in holes_rel]

    def derive_pixels_from_boundaries(self, nx, ny):
        """Reconstruct pixels from shells and holes."""

        raster = PIL.Image.new("L", [nx, ny], 0)
        draw = PIL.ImageDraw.Draw(raster)

        for shell in self.shells:
            shell = [(x, y) for x, y in shell]
            # draw.polygon(shell, fill=1, outline=1)
            PIL.ImageDraw.Draw(raster).polygon(shell, fill=1, outline=1)

        for hole in self.holes:
            hole = [(x, y) for x, y in hole]
            # draw.polygon(hole, fill=0, outline=1)
            PIL.ImageDraw.Draw(raster).polygon(hole, fill=0, outline=1)

        mask = np.array(raster, np.int8).T
        pixels = np.array(np.where(mask), np.int32).T
        values = np.zeros(int(pixels.size/2), np.float32)*np.nan

        if pixels.size != 2*self.n:
            err = (
                f"feature {self.id}: pixel reconstruction failed: "
                f"{int(pixels.size/2)} != {self.n}"
            )
            log.warning(err)
            # raise Exception(err)

        self.pixels = pixels
        self.values = values

    def area_lonlat(self, lon, lat, *, unit="km2", method="grid"):
        """Compute area of the feature in geometric distance (e.g., km^2)."""

        if lon is None:
            raise ValueError(f"lon is None (method='{method}')")
        if lat is None:
            raise ValueError(f"lat is None (method='{method}')")

        # Check units
        units = ["m2", "km2"]
        if unit not in units:
            raise ValueError(f"invalid unit '{unit}', must be one of {units}")

        if method == "grid":
            # Note: 'grid' strongly recommended over 'proj'!
            #       However, not yet implemented for non-regular grids
            if len(lon.shape) > 1:
                raise NotImplementedError(
                    f"{type(self).__name__}.area_lonlat with method 'grid' for 2D "
                    "lon/lat; if your lon/lat grid is regular, pass 1D lon/lat, "
                    "otherwise use (less precise) method 'proj'"
                )
            area_km2 = self.area_lonlat_grid(lon, lat)

        elif method == "proj":
            # Note: 'grid' is more precise than 'proj',
            #       especially for coarse grids
            #       (tested for circles on the globe)
            area_km2 = self.area_lonlat_proj(lon, lat)

        else:
            raise ValueError(f"method='{method}'")

        if unit == "m2":
            return area_km2*1e6
        elif unit == "km2":
            return area_km2

    def area_lonlat_grid(self, lon1d, lat1d):
        """Compute feature area using a grid-based approach."""
        pxs, pys = self.pixels.T
        return points_area_lonlat_reg(pxs, pys, lon1d, lat1d)

    def area_lonlat_proj(self, lon, lat):
        """Compute feature area using a projection-based approach."""

        # Sum up shell areas
        area_shells_m2 = 0.0
        for shell in self.shells:
            area_shells_m2 += self._area_lonlat_proj__core(shell, lon, lat)

        # Sum up holes areas
        area_holes_m2 = 0.0
        for hole in self.holes:
            area_holes_m2 += self._area_lonlat_proj__core(hole, lon, lat)

        # Subtract holes areas from shell areas
        area_m2 = area_shells_m2 - area_holes_m2

        # SR_TMP <
        if len(self.holes) > 0:
            log.warning(
                f"feature {self.id}: area_lonlat: subtracting area of "
                f"{len(self.holes)} holes ({area_holes_m2} m^2) from "
                f"{len(self.shells)} shells ({area_shells_m2} m^2); "
                f"holes area is slightly overestimated (include the holes' "
                f"boundary pixels which area actually part of the feature)"
            )
        # SR_TMP >

        return area_m2*1e-6

    def _area_lonlat_proj__core(self, path, lon, lat):
        """Compute the area in m**2 of a lon/lat Polygon.

        source: https://gis.stackexchange.com/a/166421

        """
        px, py = path.T
        plon = lon[px, py]
        plat = lat[px, py]

        geom = geo.Polygon([(lo, la) for lo, la in zip(plon, plat)])

        area = shapely.ops.transform(
            functools.partial(
                pyproj.transform, pyproj.Proj(init='EPSG:4326'), pyproj.Proj(
                    proj='aea', lat1=geom.bounds[1], lat2=geom.bounds[3],
                ),
            ),
            geom,
        ).area

        return area

    @classmethod
    def from_jdat(cls, jdat, name=None, pixel_tables=None,
                pixels_missing=False):
        _name_ = f"{cls.__name__}.from_jdat"

        if pixel_tables is None:
            pixels_missing = True

        # Read pixels (or provide dummy corrds/values)
        if pixel_tables is not None:
            pixels, values, shells, holes = _feature__from_jdat__pixels_from_tables(
                dict(jdat), pixel_tables, name, pixels_missing,
            )
        else:
            pixels = None
            values = None
            shells = None
            holes = None

        # Get center and extrema
        center = np.array(jdat["center"], dtype=np.int32)
        extrema = np.array(jdat["extrema"], dtype=np.int32)

        # Initialize feature
        feature = Feature(
            values=values,
            pixels=pixels,
            center=center,
            extrema=extrema,
            shells=shells,
            holes=holes,
            id_=jdat["id"],
            derive_center=not pixels_missing,
            derive_boundaries=not pixels_missing,
        )
        if pixels_missing:
            feature.hardcode_n_pixels(jdat["stats"]["n"])

        # Add properties etc.
        feature.timestep = jdat["timestep"]
        feature.properties = {key: val for key, val in jdat["properties"].items()}
        feature.associates = {key: val for key, val in jdat["associates"].items()}
        feature._stats_lst = [(key, val) for key, val in jdat["stats"].items()]

        return feature

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.float32_t distance(self, Feature other) except -999:
        cdef np.int32_t x0 = self.center[0]
        cdef np.int32_t y0 = self.center[1]
        cdef np.int32_t x1 = other.center[0]
        cdef np.int32_t y1 = other.center[1]
        return sqrt(pow(x0 - x1, 2) + pow(y0 - y1, 2))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint overlaps(self, Feature other) except -1:
        cdef np.int32_t x0
        cdef np.int32_t y0
        cdef np.int32_t x1
        cdef np.int32_t y1
        cdef int i
        cdef int j
        cdef np.int32_t[:, :] pixels0=self.pixels, pixels1=other.pixels

        # Check overlap of bounding boxes
        if not self.overlaps_bbox(other):
            return False

        # Check pixel for pixel
        for i in range(self.n):
            x0 = pixels0[i, 0]
            y0 = pixels0[i, 1]
            for j in range(other.n):
                x1 = pixels1[j, 0]
                y1 = pixels1[j, 1]
                if x0 == x1 and y0 == y1:
                    return True
        return False

    cpdef bint overlaps_bbox(self, Feature other) except -1:

        # Check of other is farther west
        cdef np.int32_t x0min = self.pixels[:, 0].min()
        cdef np.int32_t x1max = other.pixels[:, 0].max()
        if x0min > x1max:
            return False

        # Check if other is farther east
        cdef np.int32_t x0max = self.pixels[:, 0].max()
        cdef np.int32_t x1min = other.pixels[:, 0].min()
        if x0max < x1min:
            return False

        # Check if other is farther south
        cdef np.int32_t y0min = self.pixels[:, 1].min()
        cdef np.int32_t y1max = other.pixels[:, 1].max()
        if y0min > y1max:
            return False

        # Check if other is farther north
        cdef np.int32_t y0max = self.pixels[:, 1].max()
        cdef np.int32_t y1min = other.pixels[:, 1].min()
        if y0max < y1min:
            return False

        # Bounding boxes overlap
        return True

    cpdef tuple bbox(self):
        return (self.pixels[:, 0].min(), self.pixels[:, 1].min(),
                self.pixels[:, 0].max(), self.pixels[:, 1].max())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.int32_t, ndim=2] overlap_bbox(self, Feature other):
        cdef np.int32_t x0min = self.pixels[:, 0].min()
        cdef np.int32_t x1min = other.pixels[:, 0].min()
        cdef np.int32_t x0max = self.pixels[:, 0].max()
        cdef np.int32_t x1max = other.pixels[:, 0].max()
        cdef np.int32_t y0min = self.pixels[:, 1].min()
        cdef np.int32_t y0max = self.pixels[:, 1].max()
        cdef np.int32_t y1min = other.pixels[:, 1].min()
        cdef np.int32_t y1max = other.pixels[:, 1].max()
        cdef np.ndarray[np.int32_t, ndim=2] bbox
        bbox = np.empty([2, 2], dtype=np.int32)
        bbox[0, 0] = max(x0min, x1min)
        bbox[1, 0] = min(x0max, x1max)
        bbox[0, 1] = max(y0min, y1min)
        bbox[1, 1] = min(y0max, y1max)
        return bbox

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int overlap_n(self, Feature other) except -999:
        """Overlap in number of pixels."""

        # Cheap check if they overlap at all
        if not self.overlaps_bbox(other):
            return 0

        # Determine overlap of bboxes (pixels outside for sure don't overlap)
        cdef np.int32_t[:, :] bbox = self.overlap_bbox(other)

        # Check pixel for pixel (only if overlap of bboxes)
        # Looping over the bigger feature first should be faster
        # because more pixels outside the overlap bbox can be skipped
        # (the inner loop always needs to be completed)
        # SR_TODO actually measure this for confirmation!
        if self.n >= other.n:
            return self._overlap_n_core(
                self.pixels, self.n, other.pixels, other.n, bbox,
            )
        return self._overlap_n_core(other.pixels, other.n, self.pixels, self.n, bbox)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _overlap_n_core(
        self,
        np.int32_t[:, :] pixels0,
        int n0,
        np.int32_t[:, :] pixels1,
        int n1,
        np.int32_t[:, :] bbox,
    ) except -999:
        cdef np.int32_t x0
        cdef np.int32_t y0
        cdef np.int32_t x1
        cdef np.int32_t y1
        cdef int i
        cdef int j
        cdef int n = 0
        for i in prange(n0, nogil=True):
            x0 = pixels0[i, 0]
            y0 = pixels0[i, 1]
            if x0 < bbox[0, 0] or x0 > bbox[1, 0] or y0 < bbox[0, 1] or y0 > bbox[1, 1]:
                continue
            for j in range(n1):
                x1 = pixels1[j, 0]
                y1 = pixels1[j, 1]
                if x0 == x1 and y0 == y1:
                    n += 1
        return n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list find_overlapping(self, list others):
        cdef int i
        cdef int n=len(others)
        cdef Feature other
        cdef list overlapping = []
        for i in range(n):
            other = others[i]
            if self.overlaps(other):
                overlapping.append(other)
        return overlapping

    def is_mean(self):
        return False

    def is_periodic(self):
        return False

    def json_dict(self, format=True):
        """Return feature information as a json-compatible dict."""
        center = [int(self.center[0]), int(self.center[1])]
        extrema = [[int(i) for i in xy] for xy in self.extrema]
        neighbors = [neighbor.id for neighbor in self.neighbors]
        jdat = {
            "id": self.id,
            "timestep": self.timestep,
            "center": NoIndent(center) if format else center,
            "extrema": NoIndent(extrema) if format else extrema,
            "neighbors": neighbors,
        }
        self.compute_stats()
        jdat["stats"] = self.stats
        for group in ["associates", "properties"]:
            jdat[group] = {}
            for key, val in sorted(getattr(self, group).items()):
                jdat[group][key] = val
        return jdat

    def boundary_pixels(self):
        shells_pixels = [(x, y) for shell in self.shells for x, y in shell]
        holes_pixels = [(x, y) for hole in self.holes for x, y in hole]
        boundary_pixels = shells_pixels + holes_pixels
        return np.array(boundary_pixels, dtype=np.int32)

    def shared_boundary_pixels(self, key, mode="unique"):
        if mode == "unique":
            table = self._shared_boundary_pixels_unique
        elif mode == "complete":
            table = self._shared_boundary_pixels
        else:
            choices = ['unique', 'complete']
            raise ValueError(f"invalid mode '{mode}' (not among {', '.join(choices)})")

        # Pixels shared with background
        if isinstance(key, str):
            if key in ["bg", "background"]:
                return table.get("bg", [])

            # Pixels not shared (4-connectivity in convex corners)
            if key in ["in", "interior"]:
                return table.get("in", [])
        else:
            # Neighbor feature
            for neighbor in self.neighbors:
                if neighbor == key:
                    return table.get(key, [])

        raise ValueError(f"invalid key '{key}'")

    def clear_shared_boundary_pixels(self):
        self._shared_boundary_pixels = {}
        self._shared_boundary_pixels_unique = {}

    def associates_n(self, incl_empty=False):
        """Return the number of associates for each type."""
        asscs_n = []
        for name, asscs in sorted(self.associates.items()):
            if len(asscs) > 0 or incl_empty:
                asscs_n.append([name, len(asscs)])
        return asscs_n

    def id_increase(self, val):
        self.id += val

    cpdef np.ndarray to_field(
        self,
        int nx,
        int ny,
        np.float32_t bg = 0,
        bint fill_holes = False,
        bint cache=False,
    ):
        """Return feature as 2d field."""

        if cache and "field" in self._cache:
            return self._cache["field"]

        # If feature contains no pixels, reconstruct from shells
        if self.pixels is None or self.pixels.size < 2:
            if self.shells is None or len(self.shells) == 0:
                err = "feature {} has neither pixels nor shells".format(self.id)
                raise Exception(err)
            err = "feature {} has no pixels; rebuild them with {}.{}".format(
                    self.id, self.__class__.__name__,
                    "derive_pixels_from_boundaries")
            raise NotImplementedError(err)

        # SR_TODO <
        if fill_holes:
            raise NotImplemented("Feature.to_field: fill_holes == True")
        # SR_TODO >

        cdef bint has_values = (self.values.size > 0)
        cdef np.ndarray[np.float32_t, ndim=2] fld = np.full(
            [nx, ny], bg, dtype=np.float32,
        )
        cdef int i
        cdef int n = len(self.pixels)
        cdef int x
        cdef int y
        cdef np.float32_t v
        for i in range(n):
            x = self.pixels[i, 0]
            y = self.pixels[i, 1]
            if has_values:
                v = self.values[i]
            fld[x, y] = v

        if cache:
            self._cache["field"] = fld

        return fld

    cpdef np.ndarray to_mask(
        self,
        int nx,
        int ny,
        bint fill_holes = False,
        object type_ = np.bool,
        bint cache = False,
    ):
        cdef np.ndarray[np.float32_t, ndim=2] fld
        cdef np.ndarray mask

        if cache and "mask" in self._cache:
            return self._cache["mask"]

        fld = self.to_field(nx, ny, bg=np.nan, fill_holes=fill_holes, cache=cache)
        mask = ~np.isnan(fld).astype(type_)

        if cache:
            self._cache["mask"] = mask

        return mask

    cpdef object vertex(self):
        if self._vertex_name is not None:
            try:
                return self.track().graph.vs.find(self._vertex_name)
            except:
                raise Exception(f"error retrieving vertex '{self._vertex_name}'")

    cpdef void set_vertex(self, object vertex) except *:
        self._vertex_name = vertex["name"]

    cpdef void reset_vertex(self) except *:
        self._vertex_name = None

    def convex_hull(self):
        if (len(self.shells) == 0 or sum([shell.size for shell in self.shells]) == 0):
            raise Exception("cannot compute convex hull without shells")
        raise NotImplementedError("Feature.convex_hull for multiple shells") # SR_ONE_SHELL
        # hull = sp.spatial.ConvexHull(self.shell)
        # return self.shell[hull.vertices]


# :call: > --- callers ---
# :call: > stormtrack::core::identification::Feature::from_jdat
# :call: v --- calling ---
cpdef tuple _feature__from_jdat__pixels_from_tables(
    dict jdat, dict pixel_tables, str name, bint pixels_missing,
):
    cdef np.ndarray[np.float32_t, ndim=1] values
    cdef np.ndarray[np.int32_t, ndim=2] pixels
    cdef list shells
    cdef list holes

    cdef np.uint64_t pid = jdat["id"]
    cdef str pid_str = str(pid)
    if name != "":
        pid_str = name+"_"+pid_str

    # Early exit
    if pid_str in pixel_tables:
        pixels = pixel_tables[pid_str]
        values = np.array([], dtype=np.float32)
        shells = []
        holes  = []
        return pixels, values, shells, holes

    cdef str key_pixels = pid_str+"_pixels"
    cdef str key_values = pid_str+"_values"
    # SR_ONE_SHELL <
    cdef str key_shell_old = pid_str+"_shell"
    # SR_ONE_SHELL >
    cdef str key_shells_base = pid_str+"_shell_"
    cdef str key_holes_base = pid_str+"_hole_"

    # Get pixels, values
    cdef int n_pixels
    if pixels_missing:
        pixels = np.array([[]], dtype=np.int32)
        values = np.array([], dtype=np.float32)
    else:
        # Pixels
        pixels = pixel_tables[key_pixels]
        n_pixels = len(pixels)

        # Values
        if key_values not in pixel_tables:
            values = np.zeros([n_pixels], dtype=np.float32) - 1
        else:
            values = pixel_tables[key_values]

    # Get shells
    # SR_ONE_SHELL <
    cdef str key
    cdef np.ndarray[np.int32_t, ndim=2] table
    try:
        shells = [pixel_tables[key_shell_old]]
    except KeyError:
        shells = []
        for key in pixel_tables:
            if key.startswith(key_shells_base):
                table = pixel_tables[key]
                shells.append(table)
    # SR_ONE_SHELL >

    # Get holes
    holes = []
    for key in pixel_tables:
        if key.startswith(key_holes_base):
            table = pixel_tables[key]
            holes.append(table)

    return pixels, values, shells, holes


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
cpdef feature2d_from_jdat(
    dict jdat,
    np.ndarray[np.float32_t, ndim=1] values,
    np.ndarray[np.int32_t, ndim=2] pixels,
    list shells,
    list holes,
):
    """Build a feature object from a json dict and pixel data."""

    cdef np.uint64_t timestep
    cdef np.ndarray[np.int32_t, ndim=1] center
    cdef np.ndarray[np.int32_t, ndim=2] extrema

    # Get center and extrema
    timestep = jdat["timestep"]
    center = np.array(jdat["center"], dtype=np.int32)
    extrema = np.array(jdat["extrema"], dtype=np.int32)

    # Initialize feature
    cdef Feature feature = Feature(
        values=values,
        pixels=pixels,
        center=center,
        extrema=extrema,
        shells=shells,
        holes=holes,
        id_=jdat["id"],
    )
    feature.timestep = timestep

    # Add associates and properties
    cdef str group, key
    for group in ["associates", "properties"]:
        for key, val in jdat.get(group, {}).items():
            getattr(feature, group)[key] = val

    return feature


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
cpdef void features_reset_cregion(list features, bint warn=True) except *:
    if features is None:
        return
    cdef Feature feature
    for feature in features:
        feature.reset_cregion(warn)


# :call: > --- callers ---
# :call: > stormtrack::extra::front_surgery::*
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::features_find_neighbors_core
# :call: v stormtrack::core::identification::features_reset_cregion
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::constants::default_constants
# :call: v stormtrack::core::grid::grid_cleanup
# :call: v stormtrack::core::grid::grid_create
cpdef void features_find_neighbors(
    list features, Constants constants = None, np.int32_t nx = 0, np.int32_t ny = 0,
):
    # print("< features_find_neighbors")

    # Set up constants
    if constants is None:
        if nx == 0 or ny == 0:
            raise Exception("must pass either (nx, ny) or constants")
        constants = default_constants(nx=nx, ny=ny)
    else:
        nx = constants.nx
        ny = constants.ny
    cdef cConstants* cconstants = constants.to_c()

    # Initialize grid
    cdef np.ndarray[np.float32_t, ndim=2] field_raw = np.zeros(
        [cconstants.nx, cconstants.ny], dtype=np.float32,
    )
    cdef cGrid grid = grid_create(field_raw, cconstants[0])
    pixel_region_table_alloc_grid(&grid.pixel_region_table, cconstants)
    neighbor_link_stat_table_alloc(&grid.neighbor_link_stat_table, cconstants)

    features_find_neighbors_core(features, &grid, cconstants)

    # Free allocated memory
    grid_cleanup(&grid)
    features_reset_cregion(features, warn=False)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::identification::cregions2features_connected2neighbors
# :call: v stormtrack::core::identification::features_to_cregions
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::cregions::cregions_cleanup
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregions::cregions_find_connected
cdef void features_find_neighbors_core(
    list features, cGrid* grid, cConstants* constants,
) except *:
    # print("< features_find_neighbors_core")
    cdef int i
    cdef int j
    cdef int k
    cdef int n_features = len(features)

    # Turn features into cregions
    cdef cRegions cregions = cregions_create(n_features)
    cdef bint ignore_missing_neighbors = False
    features_to_cregions(
        features,
        n_features,
        &cregions,
        cregion_conf_default(),
        ignore_missing_neighbors,
        grid,
        constants,
    )

    # Find neighbors among cregions
    cdef bint reset_existing = False # SR_TODO True or False?
    cregions_find_connected(&cregions, reset_existing, constants)

    # SR_TMP <
    cdef np.uint64_t** id_table = <np.uint64_t**>malloc(n_features*sizeof(np.uint64_t*))
    cdef Feature feature
    for i in range(n_features):
        id_table[i] = <np.uint64_t*>malloc(2*sizeof(np.uint64_t))
        id_table[i][0] = cregions.regions[i].id
        feature = features[i]
        id_table[i][1] = feature.id
        # print(f"id_table[{i}] = [{cregions.regions[i].id}, {feature.id}]")
    # SR_TMP >

    # Transfer links back
    cregions2features_connected2neighbors(
        &cregions, features, id_table, grid, constants,
    )

    # SR_TMP <
    for i in range(n_features):
        free(id_table[i])
    free(id_table)
    # SR_TMP >

    # Clean up
    cregions_cleanup(&cregions, cleanup_regions=True)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::identification::cpixel2arr
# :call: v stormtrack::core::identification::cregions2features_connected2neighbors
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregions::cregions_cleanup
cdef list cregions_create_features(
    cRegions* cregions,
    np.uint64_t base_id,
    bint ignore_missing_neighbors,
    cGrid* grid,
    cConstants* constants,
    list used_ids = None,
):
    # print("< cregions_create_features")
    cdef int i
    cdef int j
    cdef cRegion* cregion
    cdef Feature feature
    cdef np.float32_t value
    cdef np.ndarray[np.float32_t, ndim=1] values
    cdef np.ndarray[np.int32_t, ndim=2] pixels
    cdef np.ndarray[np.int32_t, ndim=1] center
    cdef np.ndarray[np.int32_t, ndim=2] extrema
    cdef np.ndarray[np.int32_t, ndim=2] shell
    cdef np.ndarray[np.int32_t, ndim=2] hole
    cdef list features = []
    cdef list shells
    cdef list holes
    cdef np.uint64_t fid = base_id
    for i in range(cregions.n):
        cregion = cregions.regions[i]

        # SR_DBG_PERMANENT <
        if cregion.pixels_n == 0 or cregion.pixels_max == 0:
            log.error(
                f"warning: empty region ({cregion.pixels_n}/{cregion.pixels_max}): SKIP"
            )
            continue
        # SR_DBG_PERMANENT >

        # SR_DBG_PERMANENT <
        if not 0 <= cregion.pixels_n <= cregion.pixels_max:
            raise Exception(
                f"inconsistent cregion {cregion.id}: n={cregion.pixels_n}, "
                f"max={cregion.pixels_max}"
            )
        # SR_DBG_PERMANENT >

        values = np.zeros(cregion.pixels_n, dtype=np.float32)
        for j in range(cregion.pixels_max):
            if cregion.pixels[j] is NULL:
                continue
            values[j] = cregion.pixels[j].v
        pixels = cpixel2arr(cregion.pixels, cregion.pixels_n)

        shells = []
        for j in range(cregion.shells_n):
            shell = cpixel2arr(cregion.shells[j], cregion.shell_n[j])
            shells.append(shell)

        holes = []
        for j in range(cregion.holes_n):
            hole = cpixel2arr(cregion.holes[j], cregion.hole_n[j])
            holes.append(hole)

        if used_ids is not None:
            while fid in used_ids:
                fid += 1
        feature = Feature(
            values=values,
            pixels=pixels,
            center=None,
            extrema=None,
            shells=shells,
            holes=holes,
            id_=fid,
        )
        fid += 1
        features.append(feature)

    # Create lookup table for IDs (correspondence feature <-> cfeature)
    cdef np.uint64_t** id_table = <np.uint64_t**>malloc(cregions.n*sizeof(np.uint64_t*))
    for i in range(cregions.n):
        id_table[i] = <np.uint64_t*>malloc(2*sizeof(np.uint64_t))
        id_table[i][0] = cregions.regions[i].id
        feature = features[i]
        id_table[i][1] = feature.id
        # print(f"id_table[{i}] = [{id_table[i][0]}, {id_table[i][1]}]")

    # Transfer links back
    cregions2features_connected2neighbors(cregions, features, id_table, grid, constants)

    # Clean up ID lookup table and cregions
    for i in range(cregions.n):
        free(id_table[i])
    free(id_table)

    cregions_cleanup(cregions, cleanup_regions=True)

    return features


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregions_create_features
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::identification::cregion_find_corresponding_feature
# :call: v stormtrack::core::identification::determine_shared_boundary_pixels
# :call: v stormtrack::core::identification::initialize_surrounding_background_region
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::tables::pixel_region_table_reset
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
# need to keep track of which cregion corresponds to which feature, I guess
cdef void cregions2features_connected2neighbors(
    cRegions* cregions,
    list features,
    np.uint64_t** id_table,
    cGrid* grid,
    cConstants* constants,
) except *:
    """Turn connected regions into neighboring features."""
    # print(f"< cregions2features_connected2neighbors {cregions.n} {len(features)}")
    cdef int i
    cdef int j
    cdef int k
    cdef int n_features=len(features)
    cdef int i_feature
    cdef int i_cregion
    cdef int i_other_feature
    cdef np.uint64_t fid
    cdef cRegion* cregion
    cdef cRegion* other_cregion
    cdef Feature feature
    cdef Feature other_feature
    cdef dict source
    cdef dict target

    cdef int n_regions_max = 3
    pixel_region_table_reset(
        grid.pixel_region_table, grid.constants.nx, grid.constants.ny,
    )

    cdef cRegion cregion_bg
    cregion_init(&cregion_bg, cregion_conf_default(), cregion_get_unique_id())
    initialize_surrounding_background_region(
        cregions, &cregion_bg, grid, link_region=True,
    )

    # Loop over cregions
    for i_cregion in range(cregions.n):
        cregion = cregions.regions[i_cregion]

        # Determine shared boundary pixels
        shared_boundary_pixels = dict()
        shared_boundary_pixels_unique = dict()
        determine_shared_boundary_pixels(
            shared_boundary_pixels,
            shared_boundary_pixels_unique,
            cregion,
            &cregion_bg,
            grid,
        )

        # Find the feature corresponding to the cregion
        feature = cregion_find_corresponding_feature(
            cregion, features, cregions.n, id_table,
        )

        # Add neighbors
        feature.neighbors = []
        # print("cregion[{}].connected_n == {}".format(cregion.id, cregion.connected_n))
        for i_cregion in range(cregion.connected_n):
            other_cregion = cregion.connected[i_cregion]

            # Add feature corresponding to the connected region as neighbor
            other_feature = cregion_find_corresponding_feature(
                other_cregion, features, cregions.n, id_table,
            )

            # print(f"neighbors: {feature.id} <-> {other_feature.id}")
            feature.neighbors.append(other_feature)

            # Mode "shared"
            source = shared_boundary_pixels
            target = feature._shared_boundary_pixels
            target["bg"] = np.array(sorted(source["bg"]), dtype=np.int32)
            target["in"] = np.array(sorted(source["in"]), dtype=np.int32)
            target[other_feature] = np.array(
                sorted(source[other_cregion.id]), dtype=np.int32,
            )

            # Mode "unique"
            source = shared_boundary_pixels_unique
            target = feature._shared_boundary_pixels_unique
            target["bg"] = np.array(sorted(source["bg"]), dtype=np.int32)
            target["in"] = np.array(sorted(source["in"]), dtype=np.int32)
            target[other_feature] = np.array(
                sorted(source[other_cregion.id]), dtype=np.int32,
            )

    # SR_TODO is reset_connected necessary?
    cregion_cleanup(&cregion_bg, unlink_pixels=True, reset_connected=True)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: v --- calling ---
# :call: v stormtrack::core::identification::regiongrow_resolve_multi_assignments
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::tables::pixel_region_table_insert_region
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
# :call: v stormtrack::core::cregion::cregion_insert_pixel
cdef void determine_shared_boundary_pixels(
    dict shared_boundary_pixels,
    dict shared_boundary_pixels_unique,
    cRegion* cregion,
    cRegion* cregion_bg,
    cGrid* grid,
) except *:
    cdef bint debug = False
    if debug:
        log.debug(f"< determine_shared_boundary_pixels [{cregion.id}]")
    cdef int i
    cdef int j
    cdef int i_region
    cdef cRegion* other_cregion
    cdef tuple pixel
    cdef list shared_neighbors_i
    cdef bint interior
    cdef cPixel* cpixel
    cdef np.int32_t x
    cdef np.int32_t y
    cdef cRegion* cregion_assoc
    cdef tuple xy

    cdef dict pixels_assignments = {} # SR_TMP

    # Initialize dicts for shared pixels
    shared_boundary_pixels["bg"] = set()
    shared_boundary_pixels["in"] = set()
    shared_boundary_pixels_unique["bg"] = set()
    shared_boundary_pixels_unique["in"] = set()
    for i in range(cregion.connected_n):
        shared_boundary_pixels[cregion.connected[i].id] = set()
        shared_boundary_pixels_unique[cregion.connected[i].id] = set()

    # Initialize regions for multi-assigned and background pixels
    cdef cRegion cregion_multi, cregion_in
    cregion_init(&cregion_multi, cregion_conf_default(), cregion_get_unique_id())
    cregion_init(&cregion_in, cregion_conf_default(), cregion_get_unique_id())

    # Initialize dummy region for interior pixels, which is only a
    # placeholder dummy to be compatible with pixel_region_table
    cregion_init(&cregion_in, cregion_conf_default(), cregion_get_unique_id())

    # Collect boundary pixels
    cdef cRegion boundary_pixels
    cregion_init(&boundary_pixels, cregion_conf_default(), cregion_get_unique_id())
    for j in range(cregion.shells_n):
        for i in range(cregion.shell_n[j]):
            cregion_insert_pixel(
                &boundary_pixels,
                cregion.shells[j][i],
                link_region=False,
                unlink_pixel=False,
            )
    for j in range(cregion.holes_n):
        for i in range(cregion.hole_n[j]):
            cregion_insert_pixel(
                &boundary_pixels,
                cregion.holes[j][i],
                link_region=False,
                unlink_pixel=False,
            )

    # Determine pixel assignments
    for i in range(boundary_pixels.pixels_istart,
        boundary_pixels.pixels_iend):
        cpixel = boundary_pixels.pixels[i]
        if cpixel is NULL:
            continue
        pixel = (cpixel.x, cpixel.y)
        interior = True
        for j in range(grid.constants.n_neighbors_max):
            if grid.constants.connectivity == 4 and j%2 > 0:
                continue
            if cpixel.neighbors[j] is NULL:
                continue
            if debug:
                log.debug(
                    f"({cpixel.x}, {cpixel.y}).neighbor[{j}]: "
                    f"({cpixel.neighbors[j].x}, {cpixel.neighbors[j].y})"
                )
            if cpixel.neighbors[j].region is NULL:
                interior = False
                if debug:
                    log.debug(" -> background")
                pixel_region_table_insert_region(
                    grid.pixel_region_table, cpixel.x, cpixel.y, cregion_bg, -1,
                )
            elif cpixel.neighbors[j].region.id != cregion.id:
                interior = False
                other_cregion = cpixel.neighbors[j].region
                if debug:
                    log.debug(f" -> region {other_cregion.id}")
                pixel_region_table_insert_region(
                    grid.pixel_region_table, cpixel.x, cpixel.y, other_cregion, -1,
                )
        if interior:
            if debug:
                log.debug(" -> interior")
            pixel_region_table_insert_region(
                grid.pixel_region_table, cpixel.x, cpixel.y, &cregion_in, -1,
            )

    # Collect multi-assigned pixels
    for i_pixel in range(boundary_pixels.pixels_istart,
            boundary_pixels.pixels_iend):
        cpixel = boundary_pixels.pixels[i_pixel]
        if cpixel is NULL:
            continue
        x = cpixel.x
        y = cpixel.y
        if grid.pixel_region_table[x][y].n > 1:
            cregion_insert_pixel(
                &cregion_multi, cpixel, link_region=False, unlink_pixel=False,
            )
            # DBG_BLOCK <
            if debug:
                log.debug(
                    f" -> ({x}, {y}) multi-assigned "
                    f"({grid.pixel_region_table[x][y].n}x):"
                )
                for i in range(grid.pixel_region_table[x][y].n):
                    if grid.pixel_region_table[x][y].slots[i].region.id == cregion_bg.id:
                        rid = grid.pixel_region_table[x][y].slots[i].region.id
                        log.debug(f"  - region {rid} (background)")
                    elif grid.pixel_region_table[x][y].slots[i].region.id == cregion_in.id:
                        rid = grid.pixel_region_table[x][y].slots[i].region.id
                        log.debug(f"  - region {rid} (interior)")
                    else:
                        rid = grid.pixel_region_table[x][y].slots[i].region.id
                        log.debug(f"  - region {rid}")
            # DBG_BLOCK >

    # Insert pixels into "complete" lists
    for i_pixel in range(boundary_pixels.pixels_istart, boundary_pixels.pixels_iend):
        if boundary_pixels.pixels[i_pixel] is NULL:
            continue
        x = boundary_pixels.pixels[i_pixel].x
        y = boundary_pixels.pixels[i_pixel].y
        xy = (x, y)
        for i_slot in range(grid.pixel_region_table[x][y].n):
            cregion_assoc =  grid.pixel_region_table[x][y].slots[i_slot].region
            if cregion_assoc is NULL:
                continue
            elif cregion_assoc.id == cregion.id:
                continue
            elif cregion_assoc.id == cregion_in.id:
                if debug:
                    log.debug(f"SHARED[in] <- ({x}, {y})")
                shared_boundary_pixels["in"].add(xy)
            elif cregion_assoc.id == cregion_bg.id:
                if debug:
                    log.debug(f"SHARED[bg] <- ({x}, {y})")
                shared_boundary_pixels["bg"].add(xy)
            else:
                # DBG_PERMANENT <
                if cregion_assoc.id not in shared_boundary_pixels:
                    shared_boundary_pixels[cregion_assoc.id] = set()
                    # log.warning("warning: key {} not in shared_boundary_pixels".format(cregion_assoc.id))
                # DBG_PERMANENT >
                if debug:
                    log.debug(f"SHARED[{cregion_assoc.id}] <- ({x}, {y})")
                shared_boundary_pixels[cregion_assoc.id].add(xy)

    # Assigned to multiple targets: resolve assignment!
    # SR_TMP < SR_TODO fix this properly
    cdef np.int8_t** pixel_status_table_tmp = grid.pixel_status_table
    grid.pixel_status_table = NULL
    # SR_TMP >
    if cregion_multi.pixels_n > 0:
        regiongrow_resolve_multi_assignments(
                &cregion_multi,
                grid,
                grid.constants.n_neighbors_max,
                debug = False,
            )
    # SR_TMP <
    grid.pixel_status_table = pixel_status_table_tmp
    # SR_TMP >

    # Insert pixels into "unique" lists
    for i_pixel in range(boundary_pixels.pixels_istart,
            boundary_pixels.pixels_iend):
        if boundary_pixels.pixels[i_pixel] is NULL:
            continue
        x = boundary_pixels.pixels[i_pixel].x
        y = boundary_pixels.pixels[i_pixel].y
        xy = (x, y)
        # SR_DBG_PERMANENT <
        if grid.pixel_region_table[x][y].n != 1:
            raise Exception(
                f"determine_shared_boundary_pixels: pixel ({x}, {y}) not uniquely "
                f"assigned (n={grid.pixel_region_table[x][y].n})"
            )
        # SR_DBG_PERMANENT >
        if debug:
            rid = grid.pixel_region_table[x][y].slots[0].region.id
            rank = grid.pixel_region_table[x][y].slots[0].rank
            log.debug(f"({x}, {y}) [{rid}] {rank}")
        i_slot = 0
        cregion_assoc = grid.pixel_region_table[x][y].slots[i_slot].region
        if cregion_assoc is NULL:
            continue
        elif cregion_assoc.id == cregion.id:
            continue
        elif cregion_assoc.id == cregion_in.id:
            if debug:
                log.debug(f"UNIQUE[in] <- ({x}, {y})")
            shared_boundary_pixels_unique["in"].add(xy)
        elif cregion_assoc.id == cregion_bg.id:
            if debug:
                log.debug(f"UNIQUE[bg] <- ({x}, {y})")
            shared_boundary_pixels_unique["bg"].add(xy)
        else:
            # DBG_PERMANENT <
            if cregion_assoc.id not in shared_boundary_pixels_unique:
                shared_boundary_pixels_unique[cregion_assoc.id] = set()
                # log.warning(
                #     f"key {cregion_assoc.id} not in shared_boundary_pixels_unique"
                # )
            # DBG_PERMANENT >
            if debug:
                log.debug(f"UNIQUE[{cregion_assoc.id}] <- ({x}, {y})")
            shared_boundary_pixels_unique[cregion_assoc.id].add(xy)

    # Clean up regions (unlink background pixels)
    # SR_TODO is reset_connected necessary?
    cregion_cleanup(&cregion_in, unlink_pixels=False, reset_connected=True)
    # SR_TODO is reset_connected necessary?
    cregion_cleanup(&boundary_pixels, unlink_pixels=False, reset_connected=True)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: v --- calling ---
# :call: v stormtrack::core::identification::_find_background_neighbor_pixels
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
cdef void initialize_surrounding_background_region(
    cRegions* cregions, cRegion* cregion_bg, cGrid* grid, bint link_region,
) except *:
    cdef int mode = 1
    cdef int i_region
    cdef int i_pixel
    cdef int i_shell
    cdef int i_hole
    cdef int x
    cdef int y
    if mode == 0:
        # Only check pixels immediately surrounging the features
        for i_region in prange(cregions.n, nogil=True):
            cregion = cregions.regions[i_region]
            for i_shell in range(cregion.shells_n):
                for i_pixel in range(cregion.shell_n[i_shell]):
                        _find_background_neighbor_pixels(
                            cregion_bg,
                            cregion.shells[i_shell][i_pixel],
                            link_region,
                            grid.constants.n_neighbors_max,
                        )
            for i_hole in range(cregion.holes_n):
                for i_pixel in range(cregion.hole_n[i_hole]):
                    _find_background_neighbor_pixels(
                        cregion_bg,
                        cregion.holes[i_hole][i_pixel],
                        link_region,
                        grid.constants.n_neighbors_max,
                    )
    elif mode == 1:
        # Check the whole grid
        for x in prange(grid.constants.nx, nogil=True):
            for y in range(grid.constants.ny):
                _find_background_neighbor_pixels(
                    cregion_bg,
                    &grid.pixels[x][y],
                    link_region,
                    grid.constants.n_neighbors_max,
                )
    else:
        raise Exception(
            f"initialize_surrounging_background_region: invalid mode {mode}"
        )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::initialize_surrounding_background_region
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_insert_pixel_nogil
cdef inline void _find_background_neighbor_pixels(
    cRegion* cregion_bg, cPixel* cpixel, bint link_region, int n_neighbors_max,
) nogil:
    if cpixel is NULL:
        return
    cdef int i_neighbor
    cdef cPixel* neighbor
    if cpixel is not NULL:
        for i_neighbor in range(n_neighbors_max):
            neighbor = cpixel.neighbors[i_neighbor]
            if neighbor is not NULL:
                if neighbor.region is NULL:
                    cregion_insert_pixel_nogil(
                        cregion_bg,
                        neighbor,
                        link_region=link_region,
                        unlink_pixel=False,
                    )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
# :call: v stormtrack::core::structs::cRegion
cdef Feature cregion_find_corresponding_feature(
    cRegion* cregion, list features, int n_features, np.uint64_t** id_table,
):
    # print("< cregion_find_corresponding_feature {}".format(cregion.id))
    cdef int i
    cdef np.uint64_t fid
    cdef Feature feature
    for i in range(n_features):
        if id_table[i][0] == cregion.id:
            fid = id_table[i][1]
            break
    else:
        raise Exception(f"error: cregion id {cregion.id} not found in table")
    for i in range(n_features):
        feature = features[i]
        if feature.id == fid:
            break
    else:
        raise Exception(f"error: feature with id {fid} not found")
    # print(f"> cregion_find_corresponding_feature {cregion.id} {feature.id}")
    return feature


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregions_create_features
# :call: > stormtrack::core::identification::features_grow
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
cdef np.ndarray[np.int32_t, ndim=2] cpixel2arr(cPixel** pixels, int n):
    """Convert an cPixel C-array into a numpy array."""
    cdef int i
    cdef np.ndarray[np.int32_t, ndim=2] arr = np.empty([n, 2], dtype=np.int32)
    for i in range(n):
        arr[i, 0] = pixels[i][0].x
        arr[i, 1] = pixels[i][0].y
    return arr


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
def features_neighbors_id2obj(features, missing_action="error"):
    _name_ = "features_neighbors_id2obj"
    features_fid = {f.id: f for f in features}
    for feature in features:
        neighbors = []
        for neighbor in feature.neighbors:
            if isinstance(neighbor, Feature):
                neighbors.append(neighbor)
            elif neighbor in features_fid:
                neighbors.append(features_fid[neighbor])
            else:
                if missing_action == "ignore":
                    continue
                msg = f"{_name_}: feature {feature.id}: missing neighbor {neighbor}"
                if missing_action == "warn":
                    log.warning(msg)
                elif missing_action == "error":
                    raise Exception(msg)
        feature.neighbors = neighbors


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
def features_neighbors_obj2id(features, names=None):
    if isinstance(features, dict):
        for feature_name, features in features.items():
            if names is not None and feature_name not in names:
                continue
            for feature in features:
                feature.neighbors = [
                    (f.id if isinstance(f, Feature) else f) for f in feature.neighbors
                ]
    else:
        if names is not None:
            raise ValueError("name not None, but features not in named form in a dict")
        for feature in features:
            feature.neighbors = [
                (f.id if isinstance(f, Feature) else f) for f in feature.neighbors
            ]


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
cpdef void associate_features(
    str name1,
    list features1,
    str name2,
    list features2,
    bint keep_existing = True,
    bint check_timesteps = True,
) except *:

    cdef Feature feature
    cdef Feature feature1
    cdef Feature feature2
    cdef int i
    cdef int i1
    cdef int i2
    cdef int n1 = len(features1)
    cdef int n2 = len(features2)

    # Check that all features have the same timestep
    cdef np.uint64_t timestep
    if check_timesteps:
        if len(features1) > 0:
            timestep = features1[0].timestep
        else:
            timestep = features2[0].timestep
        if check_timesteps:
            for i in range(n1):
                feature = features1[i]
                if feature.timestep != timestep:
                    raise Exception(
                        f"inconsistent timesteps: {feature.timestep} != {timestep}"
                    )
            for i in range(n2):
                feature = features2[i]
                if feature.timestep != timestep:
                    raise Exception(
                        f"inconsistent timesteps: {feature.timestep} != {timestep}"
                    )

    # Remove existing associates
    if not keep_existing:
        for i in range(n1):
            feature = features1[i]
            if name2 in feature.associates:
                feature.associates[name2] = []
        for i in range(n2):
            feature = features2[i]
            if name1 in feature.associates:
                feature.associates[name1] = []

    # If one of the sets is empty, we're done
    if n1 == 0 or n2 == 0:
        return

    # Associate features
    cdef list associates
    for i1 in range(n1):
        feature1 = features1[i1]
        if name2 not in feature1.associates:
            feature1.associates[name2] = []
        associates = feature1.find_overlapping(features2)
        for i2 in range(n2):
            feature2 = features2[i2]
            if name1 not in feature2.associates:
                feature2.associates[name1] = []
            if feature2 in associates:
                if feature2.id not in feature1.associates[name2]:
                    feature1.associates[name2].append(feature2.id)
                if feature1.id not in feature2.associates[name1]:
                    feature2.associates[name1].append(feature1.id)


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
cpdef int resolve_indirect_associations(
    str name1, list features1, str name2, list features2, str name3, list features3,
) except -1:
    cdef dict features2_table = {f.id: f for f in features2}
    cdef dict features3_table = {f.id: f for f in features3}
    cdef int i
    cdef int n
    cdef int n_fids2
    cdef int n_fids3
    cdef int i_feature1
    cdef int n_features1=len(features1)
    cdef int i_feature2
    cdef int n_features2=len(features2)
    cdef int i_feature3
    cdef int n_features3=len(features3)
    cdef Feature feature1
    cdef Feature feature2
    cdef Feature feature3
    cdef dict associates1
    cdef dict associates2
    cdef dict associates3
    cdef list fids2
    cdef list fids3
    cdef np.uint64_t fid1
    cdef np.uint64_t fid2
    cdef np.uint64_t fid3
    cdef str assoc_name13
    cdef str assoc_name_31
    cdef list assoc_ids_13
    cdef list assoc_ids_31
    cdef list ids
    assoc_name_13 = name2+"/"+name3
    assoc_name_31 = name2+"/"+name1

    # Loop over 'source' features
    cdef int n_assoc = 0
    for i_feature1 in range(n_features1):
        feature1 = features1[i_feature1]
        fid1 = feature1.id
        if name2 not in feature1.associates:
            continue

        # Loop over associated 'intermediary' features
        fids2 = feature1.associates[name2]
        n_fids2 = len(fids2)
        for i_feature2 in range(n_fids2):
            fid2 = fids2[i_feature2]
            feature2 = features2_table[fid2]
            if name3 not in feature2.associates:
                continue

            # Loop over associated 'target' features
            fids3 = feature2.associates[name3]
            n_fids3 = len(fids3)
            for i_feature3 in range(n_fids3):
                fid3 = fids3[i_feature3]
                feature3 = features3_table[fid3]

                # Connect 'source' to 'target' feature
                assoc_ids_13 = [fid2, fid3]
                if assoc_name_13 not in feature1.associates:
                    feature1.associates[assoc_name_13] = []
                ids = feature1.associates[assoc_name_13]
                for i in range(len(ids)):
                    if (ids[i][0] == assoc_ids_13[0] and
                            ids[i][1] == assoc_ids_13[1]):
                        break
                else:
                    n_assoc += 1
                    # print(f"feature1.associates[{assoc_name_13}].append({assoc_ids_13})")
                    feature1.associates[assoc_name_13].append(assoc_ids_13)

                # Connect 'target' to 'source' feature
                assoc_ids_31 = [fid2, fid1]
                if assoc_name_31 not in feature3.associates:
                    feature3.associates[assoc_name_31] = []
                ids = feature3.associates[assoc_name_31]
                for i in range(len(ids)):
                    if (ids[i][0] == assoc_ids_31[0] and
                            ids[i][1] == assoc_ids_31[1]):
                        break
                else:
                    n_assoc += 1
                    # print(
                    #     f"feature3.associates[{assoc_name_31}].append({assoc_ids_31})"
                    # )
                    feature3.associates[assoc_name_31].append(assoc_ids_31)

    return n_assoc


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::identification::Feature
def features_associates_obj2id(features, names=None):
    """Replace feature objects in associated features by feature ids."""
    if isinstance(features, dict):
        for feature_name, features in features.items():
            if names is not None and feature_name not in names:
                continue
            for feature in features:
                for assc_name, assc_features in feature.associates.items():
                    feature.associates[assc_name] = [
                        (f.id if isinstance(f, Feature) else f) for f in assc_features
                    ]
    else:
        if names is not None:
            err = ("argument names meaninless unless features are passed "
                    "in named form in a dict")
            raise ValueError(err)
        for feature in features:
            for assc_name, assc_features in feature.associates.items():
                feature.associates[assc_name] = [
                    (f.id if isinstance(f, Feature) else f) for f in assc_features
                ]


# :call: > --- callers ---
# :call: v --- calling ---
def features_associates_id2obj(
    features_named,
    names=None,
    missing_ids=None,
    missing_action="error",
    missing_verbose=True,
):
    """Replace feature ids in associated features by feature objects."""
    if missing_ids is None:
        missing_ids = {}
    if missing_action not in ["error", "warning", "ignore"]:
        raise ValueError(f"invalid missing_action '{missing_action}'")

    # Replace feature ids of associated features by feature objects
    for feature_name, features in features_named.items():
        if names is not None and feature_name not in names:
            continue
        for feature in features:
            for assoc_name, fids in feature.associates.items():
                if names is not None and assoc_name not in names:
                    continue
                if assoc_name not in features_named:
                    continue
                others = features_named[assoc_name]
                others_fids = [f.id for f in others]
                for fid in fids.copy():
                    try:
                        other = [f for f in others if f.id == fid][0]
                    except IndexError:
                        err = f"associated {assoc_name} feature not found: {fid} "
                        if assoc_name not in missing_ids:
                            missing_ids[assoc_name] = set()
                        if fid not in missing_ids[assoc_name]:
                            # Handle missing feature
                            missing_ids[assoc_name].add(fid)
                            if missing_action == "error":
                                raise Exception(err) from None
                            elif missing_action == "warning":
                                if missing_verbose:
                                    log.warning(err)
                            elif missing_action == "ignore":
                                pass
                    else:
                        feature.associates[assoc_name].append(other)
                    feature.associates[assoc_name].remove(fid)

    # Handle missing features
    if len(missing_ids) > 0:
        if missing_action == "warning":
            if len(missing_ids) == 1:
                name, fids = next(iter(missing_ids.items()))
                log.warning(f"{name} associated {len(fids)} features not found")
            else:
                n = sum([len(i) for i in missing_ids.values()])
                s_missing_ids = ", ".join(
                    [
                        "{}*{}".format(len(ids), name)
                        for name, ids in sorted(
                            missing_ids.items(), key=lambda i: len(i[1])
                        )
                    ]
                )
                log.warning("{n} associated features not found: {s_missing_ids}")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cyclones_to_features
# :call: v --- calling ---
def oldfeature_to_pixels(oldfeature, lon, lat, vb=True):
    """Extract pixels, center, and extrema from old-style cyclone feature."""

    pixels_features_ts = {}

    # Get the path of the outermost contour
    if not oldfeature.is_mean():
        paths = [oldfeature.path()]
    else:
        paths = [f.path() for f in oldfeature.features()]

    # Get all pixels (compute the kdtree only once)
    mask, kdtree = paths_lonlat_to_mask(paths, lon, lat, return_tree=True)
    pixels = np.dstack(np.where(mask > 0))[0]

    def _get_indices(xy, tree, shape):
        _, ind = tree.query(xy)
        return np.array(np.unravel_index(ind, shape))

    # Get the center, minima, and shell
    center = _get_indices(oldfeature.center(), kdtree, lon.shape)
    extrema = np.array(
        [_get_indices(pt.xy, kdtree, lon.shape) for pt in oldfeature.minima()]
    )
    shell = np.array(
        [_get_indices(xy, kdtree, lon.shape) for xy in oldfeature.path()]
    )

    return (
        pixels.astype(np.int32),
        extrema.astype(np.int32),
        center.astype(np.int32),
        shell.astype(np.int32)
    )


# :call: > --- callers ---
# :call: > stormtrack::identify_features::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::oldfeature_to_pixels
# :call: v stormtrack::core::identification::Feature
def cyclones_to_features(ts, cyclones, slp, lon, lat, vb=True, out=None):
    """Convert old-style cyclone features into new-style Feature objects."""

    if out is None:
        out = []

    for cyclone in cyclones:
        if vb:
            print(f"convert cyclone {cyclone.id()}")

        # Extract feature pixels
        pixels, extrema, center, shell = oldfeature_to_pixels(cyclone, lon, lat)
        if vb:
            print(f"extracted {len(pixels)} pixels, {len(extrema)} extrema, and center")

        # Turn pixels into new-style features
        if vb:
            print("create Feature object")
        values = slp[pixels[:, 0], pixels[:, 1]]
        feature = Feature(
            pixels=pixels,
            values=values,
            center=center,
            extrema=extrema,
            shells=[shell],
            id_=cyclone.id(),
            timestep=ts
        )
        feature.timestep = ts

        if out is not None:
            out.append(feature)

    return out
