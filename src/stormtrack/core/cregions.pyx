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
# :call: > stormtrack::core::cregions::categorize_boundaries
# :call: v --- calling ---
cdef inline int sign(int num):
    if num >= 0:
        return 1
    else:
        return -1


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
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_check_validity
# :call: v stormtrack::core::cregions::boundary_must_be_a_shell
# :call: v stormtrack::core::cregions::categorize_boundary_is_shell
# :call: v stormtrack::core::cregions::cregions_find_northernmost_uncategorized_region
# :call: v stormtrack::core::cpixel::cpixel_angle_to_neighbor
# :call: v stormtrack::core::cregion::sign
cdef bint* categorize_boundaries(cRegions* boundaries, cGrid* grid) except *:
    cdef bint debug = False
    if debug:
        log.debug(f"< categorize_boundaries: {boundaries.n}")
    cdef int ib
    cdef int ib_sel
    cdef int ic
    cdef int dx
    cdef int dy
    cdef int da
    cdef int n_bnds = boundaries.n
    cdef int n_pixels
    cdef bint* boundary_is_shell = <bint*>malloc(n_bnds*sizeof(bint))
    cdef bint* categorized = <bint*>malloc(n_bnds*sizeof(bint))
    for ib in range(n_bnds):
        categorized[ib] = False
    cdef cRegion* boundary
    cdef cPixel* cpixel
    cdef cPixel* cpixel_sel
    cdef cPixel* cpixel_pre2
    cdef cPixel* cpixel_pre
    for ib in range(n_bnds):
        if categorized[ib]:
            continue
        boundary = boundaries.regions[ib]
    #
    # Algorithm:
    #
    # - Find the uncategorized boundary with the northernmost pixel.
    #   It must be a shell, because all other boundaries are either contained
    #   by it (holes or nested shells), or other shells located further south.
    #
    # - Mark all pixels inside this shell, and collect all uncategorized
    #   boundaries comprised of such pixels (one match is sufficient, but if
    #   one pixel matches, then all should; check that in debug mode).
    #
    # - Among these boundaries, select the northernmost, which (by the same
    #   logic as before) must be a hole. Then mark all contained pixels and
    #   select all nested boundaries. Repeat until there are no more nested
    #   boundaries.
    #
    # - Repeat until all boundaries have been assigned.
    #
    cdef int* d_angles = NULL
    cdef int* angles = NULL
    cdef int i_a
    cdef int n_a
    cdef int i_da
    cdef int n_da
    cdef int i_i
    cdef int n_i
    cdef int* px_inds = NULL
    cdef int angle_pre
    cdef int n_pixels_eff
    cdef int iter_i
    cdef int iter_max=10000
    for iter_i in range(iter_max):
        if debug:
            log.debug(f"  ----- ITER {iter_i:,}/{iter_max:,} -----")

        # Select unprocessed boundary with northernost pixel
        ib_sel = cregions_find_northernmost_uncategorized_region(boundaries, categorized)
        if ib_sel < 0:
            # All boundaries categorized!
            if debug:
                log.debug(f"  ----- DONE {iter_i}/{iter_max} -----")
            break
        boundary = boundaries.regions[ib_sel]
        cregion_check_validity(boundary, ib_sel)
        n_pixels = boundary.pixels_n
        if debug:
            log.debug(f"  process boundary {ib_sel}/{n_bnds} ({n_pixels} px)")

        # SR_DBG_NOW <
        # - print(f"\n boundary # {ib_sel} ({n_pixels}):")
        # - for i in range(n_pixels):
        # -     print(f" {i:2} ({boundary.pixels[i].x:2},{boundary.pixels[i].y:2})")
        # SR_DBG_NOW >

        d_angles = <int*>malloc(n_pixels*sizeof(int))
        angles   = <int*>malloc(n_pixels*sizeof(int))
        px_inds  = <int*>malloc(n_pixels*sizeof(int))
        for i in range(n_pixels):
            d_angles[i] = 999
            angles  [i] = 999
            px_inds [i] = 999
        i_a = -1
        n_a = 0
        i_da = -1
        n_da = 0
        i_i = -1
        n_i = 0

        # - print(f"<<<<< ({n_pixels})") # SR_DBG_NOW

        # Determine the sense of rotation of the boundary
        cpixel_pre = boundary.pixels[n_pixels - 2]
        angle_pre = -999
        n_pixels_eff = n_pixels
        for i_px in range(n_pixels):

            # Check in advance if boundary must be a shell to prevent issues
            if boundary_must_be_a_shell(n_pixels_eff, grid):
                boundary_is_shell[ib_sel] = True
                categorized[ib_sel] = True
                if debug:
                    log.debug(
                        f"  boundary # {ib_sel} too short ({n_pixels} px) "
                        f"to be a hole ({grid.constants.connectivity}-connectivity)"
                    )
                break

            i_i += 1
            n_i += 1
            px_inds[i_i] = i_px
            cpixel = boundary.pixels[i_px]
            # - print('# ', i_px, px_inds[i_a], (i_i, n_i), (i_a, n_a), (i_da, n_da), (cpixel.x, cpixel.y)) # SR_DBG_NOW

            # Determine angle from previous to current pixel
            angle = cpixel_angle_to_neighbor(cpixel, cpixel_pre)
            i_a += 1
            n_a += 1
            angles[i_a] = angle
            px_inds[i_a] = i_px
            # - print((cpixel_pre.x, cpixel_pre.y), (cpixel.x, cpixel.y), angle)

            # SR_DBG <
            # - lst1 = [int(px_inds[i]) for i in range(min(n_a, n_pixels))]
            # - lst2 = [int(d_angles[i]) for i in range(min(n_da, n_pixels))]
            # - print(f"> {int(i_a)} {int(n_a)} {lst1}")
            # - print(f"> {int(i_da)} {int(n_da)} {lst2}")
            # SR_DBG >

            if angle_pre != -999:
                da = angle - angle_pre
                if abs(da) > 180:
                    da = da - sign(da) * 360
                i_da += 1
                n_da += 1
                d_angles[i_da] = da

                if da == 180:
                    # Change to opposite direction indicates an isolated
                    # boundary pixel, which can simply be ignored
                    # SR_TMP <
                    if i_i < 2:
                        raise NotImplementedError(f"i_i: {i_i} < 2")
                    # SR_TMP >
                    cpixel_pre2 = boundary.pixels[px_inds[i_i - 2]]

                    # print(
                    #     f"! {i_i - 2:2}/{px_inds[i_i - 2]:2}"
                    #     f"({cpixel_pre2.x:2},{cpixel_pre2.y:2})"
                    #     f" {i_i - 1:2}/{px_inds[i_i - 1]:2}"
                    #     f"({cpixel_pre1.x:2},{cpixel_pre.y:2})"
                    #     f" {i_i - 0:2}/{px_inds[i_i - 0]:2}"
                    #     f"({cpixel.x:2},{cpixel.y:2})"
                    # )
                    # raise Exception("ambiguous angle: 180 or -180?")

                    if cpixel_pre2.x != cpixel.x or cpixel_pre2.y != cpixel.y:
                        exit(4)

                    # SR_DBG_NOW <
                    # - print()
                    # - for i in range(n_a): print(i, i_a, n_a, angles[i])
                    # - print()
                    # - for i in range(n_da): print(i, i_da, n_da, d_angles[i])
                    # - print()
                    # - for i in range(n_i): print(i, i_i, n_i, px_inds[i])
                    # - print()
                    # SR_DBG_NOW >

                    i_i  = max(i_i  - 2, -1)
                    n_i  = max(n_i  - 2,  0)
                    i_da = max(i_da - 2, -1)
                    n_da = max(n_da - 2,  0)
                    i_a  = max(i_a  - 2, -1)
                    n_a  = max(n_a  - 2,  0)
                    n_pixels_eff -= 2
                    angle_pre = angles[i_a]
                    cpixel_pre = cpixel

                    # SR_DBG_NOW <
                    # - print()
                    # - for i in range(n_a): print(i, i_a, n_a, angles[i])
                    # - print()
                    # - for i in range(n_da): print(i, i_da, n_da, d_angles[i])
                    # - print()
                    # - for i in range(n_i): print(i, i_i, n_i, px_inds[i])
                    # - print()
                    # - print((i_i, n_i), (i_da, n_da), (i_a, n_a), n_pixels_eff, angle_pre)
                    # - print(" OK cpixel_pre2 == cpixel")
                    # SR_DBG_NOW >
                    continue
                # SR_TMP >

                # DBG_BLOCK <
                if debug:
                    log.debug(
                        f"  {ic:2} ({cpixel_pre.x:2}, {cpixel_pre.y:2}) "
                        f"-> ({cpixel.x:2}, {cpixel.y:2}) {angle_pre:4} "
                        f"-> {angle:4} : {da:4}"
                    )
                # DBG_BLOCK >

            cpixel_pre = cpixel
            angle_pre = angle

        if not categorized[ib_sel]:
            categorize_boundary_is_shell(ib_sel, d_angles, n_da, boundary_is_shell)
            categorized[ib_sel] = True

        free(angles)
        free(d_angles)
        free(px_inds)
    else:
        raise Exception(
            f"categorize_boundaries: timed out after {iter_max:,} iterations"
        )

    # Clean up
    free(categorized)

    if debug:
        log.debug("> categorize_boundaries: done")

    return boundary_is_shell


# :call: > --- callers ---
# :call: > stormtrack::core::cregions::categorize_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
cdef bint boundary_must_be_a_shell(int n_pixels_eff, cGrid* grid):
    """Check whether a boundary is too short to be a hole.

    Hole boundaries have a minimum length of four pixels for 4-connectivity and
    eight pixels for 8-connectivity, respectively, plus one because the first
    pixel is contained twice. Any shorter boundary must be a shell because it
    is too short to enclose a hole!

    This may be checked explicitly in advance to avoid problems that may be
    caused by such short boundaries further down the line.

    """
    if grid.constants.connectivity == 4:
        return n_pixels_eff < 9
    elif grid.constants.connectivity == 8:
        return n_pixels_eff < 5
    else:
        log.error(f"invalid connectivity: {grid.constants.connectivity}")
        exit(1)


# :call: > --- callers ---
# :call: > stormtrack::core::cregions::categorize_boundaries
# :call: v --- calling ---
cdef void categorize_boundary_is_shell(
    int ib_sel, int* d_angles, int n_da, bint* boundary_is_shell,
):
    # Sum up the angles
    cdef int da_sum = 0
    cdef int i_da
    for i_da in range(n_da):
        # - print(f" {i_da:2} {d_angles[i_da]:4} {int(da_sum):4}")
        da_sum += d_angles[i_da]
    # print(f"       {int(da_sum):4}")

    # Categorize the boundary
    if da_sum == -360:
        boundary_is_shell[ib_sel] = True
        # print(">"*5+" SHELL")
    elif da_sum == 360:
        boundary_is_shell[ib_sel] = False
        # print(">"*5+" SHELL")
    else:
        raise Exception(
            f"categorization of boundary # {ib_sel} failed: "
            f"total angle {da_sum} != +-360"
        )


# :call: > --- callers ---
# :call: > stormtrack::core::cregions::categorize_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_northernmost_pixel
cdef int cregions_find_northernmost_uncategorized_region(
    cRegions* boundaries, bint* categorized,
):
    cdef cPixel* cpixel = NULL
    cdef cPixel* cpixel_sel = NULL
    cdef int ib_sel = -1
    cdef int ib
    for ib in range(boundaries.n):
        if not categorized[ib]:
            cpixel = cregion_northernmost_pixel(boundaries.regions[ib])
            if (
                cpixel_sel is NULL
                or cpixel.y > cpixel_sel.y
                or cpixel.y == cpixel_sel.y and cpixel.x < cpixel_sel.x
            ):
                cpixel_sel = cpixel
                ib_sel = ib
    return ib_sel


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
