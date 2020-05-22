# !/usr/bin/env python3

from __future__ import print_function

# C: C libraries
from libc.stdlib cimport exit
from libc.stdlib cimport free
from libc.stdlib cimport malloc

# C: Third-party
cimport cython
cimport numpy as np
from cython.parallel cimport prange

# # Standard library
import logging as log

# Third-party
import numpy as np


# :call: > --- callers ---
# :call: > stormtrack::core::cregions::cregions_create
# :call: v --- calling ---
cdef np.uint64_t cregions_get_unique_id():
    global CREGIONS_NEXT_ID
    cdef np.uint64_t rid = CREGIONS_NEXT_ID
    CREGIONS_NEXT_ID += 1
    return rid


# :call: > --- callers ---
# :call: > stormtrack::core::cregions::cregions_create
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegions
cdef void cregions_init(cRegions* cregions):
    cregions.id = 99999999
    cregions.n = 0
    cregions.max = 0
    cregions.regions = NULL


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::cregions_create_features
# :call: > stormtrack::core::identification::cregions_merge_connected
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_cleanup
cdef void cregions_cleanup(cRegions* cregions, bint cleanup_regions):
    cdef bint debug = False
    if debug:
        log.debug(f"< cregions_cleanup [{cregions.id}]")
    cdef int i
    if cleanup_regions:
        if debug:
            log.debug(f" -> clean up regions (n={cregions.n}, max={cregions.max})")
        for i in range(cregions.n):
            # SR_DBG <
            if cregions.regions[i] is NULL:
                log.error(f"error: cregions_cleanup: cregions {i} is NULL")
                exit(4)
            # SR_DBG >
            cregion_cleanup(
                cregions.regions[i], unlink_pixels=True, reset_connected=False,
            )
    if cregions.max > 0:
        free(cregions.regions)
        cregions.regions = NULL
        cregions.n = 0
        cregions.max = 0
    # if debug:
    #     log.debug("> cregions_cleanup")


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::_cregion_reset_connected
# :call: v stormtrack::core::cregion::cregion_connect_with
# :call: v stormtrack::core::cregions::_cregions_check_connected
cdef void cregions_find_connected(
    cRegions* cregions, bint reset_existing, cConstants* constants,
) except *:
    cdef debug = False
    cdef str _name_ = "cregions_find_connected"
    if debug:
        log.debug(f"< {_name_}: n={cregions.n}")
    cdef int i_region
    cdef int i_pixel
    cdef int i_neighbor
    cdef int i_connected
    cdef cRegion* cregion
    cdef cPixel* cpixel
    cdef cPixel* neighbor

    # Clear existing connections
    if reset_existing:
        for i_region in range(cregions.n):
            _cregion_reset_connected(cregions.regions[i_region], unlink=False)

    # DBG_BLOCK <
    if debug:
        log.debug(f"=== {cregions.n}")
        for i_region in range(cregions.n):
            cregion = cregions.regions[i_region]
            log.debug(f"({i_region}/{cregions.n}) {cregion.id}")
            if cregion.connected_n > 0:
                log.debug(
                    f" !!! cregion {cregion.id} connected_n={cregion.connected_n}"
                )
                for i_neighbor in range(cregion.connected_n):
                    log.debug(f"    -> {cregion.connected[i_neighbor].id}")
        log.debug("===")
    # DBG_BLOCK >

    _cregions_check_connected(cregions, _name_+"(0)") # SR_DBG_PERMANENT

    cdef int i_shell = 0
    for i_region in range(cregions.n):
        cregion = cregions.regions[i_region]
        if debug:
            log.debug(
                f"[{cregion.id}] {cregion.pixels_n} pixels, "
                f"{cregion.shell_n[i_shell]} shell pixels"
            )

        for i_pixel in range(cregion.shell_n[i_shell]):
            cpixel = cregion.shells[i_shell][i_pixel]
            if debug:
                log.debug(
                    f" ({cpixel.x:2},{cpixel.y:2})"
                    f"[{'-' if cpixel.region is NULL else cpixel.region.id}]"
                )

            # SR_DBG_PERMANENT <
            if cpixel.region.id != cregion.id:
                log.error(
                    f"[{_name_}] error: shell pixel ({cpixel.x}, {cpixel.y}) of region "
                    f"{cregion.id} connected to wrong region {cpixel.region.id}"
                )
                with cython.cdivision(True):
                    i_pixel = 5 / 0  # force abort
                exit(4)
            # SR_DBG_PERMANENT >

            for i_neighbor in range(constants.n_neighbors_max):
                neighbor = cpixel.neighbors[i_neighbor]
                if neighbor is NULL:
                    continue
                if neighbor.region is NULL:
                    continue
                if neighbor.region.id == cpixel.region.id:
                    continue
                for i_region2 in range(cregions.n):
                    if neighbor.region.id == cregions.regions[i_region2].id:
                        break
                else:
                    continue
                for i_connected in range(cregion.connected_n):
                    if cregion.connected[i_connected].id == neighbor.region.id:
                        # neighbor region already found previously
                        if debug:
                            log.debug(
                                f" -> already connected {cregion.id} <-> "
                                f"{neighbor.region.id}"
                            )
                        break
                else:
                    # print(f" -> connect {cregion.id} <-> {neighbor.region.id}")
                    cregion_connect_with(cregion, neighbor.region)

    _cregions_check_connected(cregions, _name_+"(1)")  # SR_DBG_PERMANENT


# :call: > --- callers ---
# :call: > stormtrack::core::cregions::cregions_find_connected
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# SR_DBG_PERMANENT
cdef void _cregions_check_connected(cRegions* cregions, str msg) except *:
    # print(f"dbg_check_connected_old {cregions.n} {msg}")
    cdef int i
    cdef int j
    cdef int k
    cdef cRegion* cregion_i
    cdef cRegion* cregion_j
    for i in range(cregions.n):
        cregion_i = cregions.regions[i]
        for j in range(cregion_i.connected_n):
            cregion_j = cregion_i.connected[j]
            if cregion_i.id == cregion_j.id:
                log.error(
                    f"[dbg_check_connected_old:{msg}] error: "
                    f"cregion {cregion_i.id} connected to itself"
                )
                exit(8)
            for k in range(cregions.n):
                if k == i:
                    continue
                if cregion_j.id ==  cregions.regions[k].id:
                    break
            else:
                log.error(
                    f"[dbg_check_connected_old:{msg}] error: cregion {cregion_i.id} "
                    f"connected to missing cregion {cregion_j.id}"
                )
                exit(8)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::cregions_merge_connected
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
# :call: > stormtrack::core::tracking::FeatureTracker::_find_successor_candidate_combinations
# :call: > stormtrack::core::tracking::FeatureTracker::extend_tracks
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregions::cregions_get_unique_id
# :call: v stormtrack::core::cregions::cregions_init
cdef cRegions cregions_create(int nmax):
    cdef bint debug = False
    cdef np.uint64_t rid = cregions_get_unique_id()
    if debug:
        log.debug(f"< cregions_create_old [{rid}] n={nmax}")
    cdef int i
    cdef cRegions cregions
    cregions_init(&cregions)
    cregions.id = rid
    cregions.n = 0
    cregions.max = nmax
    cregions.regions = <cRegion**>malloc(nmax*sizeof(cRegion*))
    for i in range(cregions.max):
        cregions.regions[i] = NULL
    return cregions


# :call: > --- callers ---
# :call: > stormtrack::core::cregions::cregions_link_region
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
cdef void cregions_extend(cRegions* cregions):
    cdef int i
    cdef int nmax_old
    cdef int nmax_new
    nmax_old = cregions.max
    nmax_new = 5*nmax_old
    if nmax_old == 0:
        nmax_new = 5

    # Store pointers to regions in temporary array
    cdef cRegion** tmp
    if nmax_old > 0:
        tmp = <cRegion**>malloc(nmax_old*sizeof(cRegion*))
        for i in prange(nmax_old, nogil=True):
            tmp[i] = cregions.regions[i]
        free(cregions.regions)

    # Allocate larger pointer array and transfer pointers back
    cregions.regions = <cRegion**>malloc(nmax_new*sizeof(cRegion*))
    if nmax_old > 0:
        for i in prange(nmax_old, nogil=True):
            cregions.regions[i] = tmp[i]
        free(tmp)
    for i in prange(nmax_old, nmax_new, nogil=True):
        cregions.regions[i] = NULL
    cregions.max = nmax_new


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregions_merge_connected_inplace
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegions
cdef void cregions_move(cRegions* source, cRegions* target):
    cdef bint debug = False
    if debug:
        log.debug("< cregions_move_old")
    if target.regions is not NULL:
        free(target.regions)
    target.regions = source.regions
    source.regions = NULL
    target.n = source.n
    target.max = source.max
    source.n = 0
    source.max = 0


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_reset
cdef void cregions_reset(cRegions* cregions):
    # print("< cregions_reset")
    cdef int i_region
    for i_region in range(cregions.n):
        cregion_reset(
            cregions.regions[i_region], unlink_pixels=True, reset_connected=False,
        )
    cregions.n = 0


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
# :call: > stormtrack::core::tracking::FeatureTracker::_find_successor_candidates
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregions::cregions_extend
cdef void cregions_link_region(
    cRegions* cregions, cRegion* cregion, bint cleanup, bint unlink_pixels,
):
    cdef bint debug = False
    if debug:
        log.debug(f"< cregions_link_region {cregion.id}")
    cdef int i
    cdef int j
    cdef int n

    if cregions.max == 0 or cregions.n >= cregions.max:
        cregions_extend(cregions)

    n = cregions.n
    if cleanup:
        # SR_TODO is reset_connected necessary?
        cregion_cleanup(cregions.regions[n], unlink_pixels, reset_connected=True)
    cregions.regions[n] = cregion
    cregions.n += 1
