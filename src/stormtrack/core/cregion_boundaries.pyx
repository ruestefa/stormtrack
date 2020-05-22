# -*- coding: utf-8 -*-

from __future__ import print_function

# C: C libraries
from libc.stdlib cimport free
from libc.stdlib cimport malloc

# C: Third-party
cimport cython
cimport numpy as np

# Standard library
import logging as log

# Third-party
import numpy as np


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregion_boundaries::cregions_determine_boundaries
cdef void cregion_determine_boundaries(cRegion* cregion, cGrid* grid) except *:
    cdef cRegions cregions = cregions_create(1)
    cregions.n = 1
    cregions.regions[0] = cregion
    cregions_determine_boundaries(&cregions, grid)
    free(cregions.regions)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: v stormtrack::core::cregion::cregion_reset_boundaries
cdef void cregions_determine_boundaries(cRegions* cregions, cGrid* grid) except *:
    cdef bint debug = False
    cdef bint debug_dump = True
    if debug:
        log.debug("< cregions_determine_boundaries")
    cdef int i_region
    cdef cRegion* cregion

    if debug:
        log.debug("reset boundaries")
    for i_region in range(cregions.n):
        cregion = cregions.regions[i_region]
        if debug:
            log.debug(" -> region {cregion.id}")
        cregion_reset_boundaries(cregion)

    cdef int i  # SR_DBG
    if debug:
        log.debug("determine new boundaries")
    cdef int n_empty=0
    for i_region in range(cregions.n):
        cregion = cregions.regions[i_region]
        if debug:
            log.debug(f" -> region {i_region} ({cregion.pixels_n} pixels)")
        if cregion.pixels_n < 1:
            log.warning(f"cregion {cregion.id} empty")
            n_empty += 1
            continue
        # SR_DBG < disable intermediate catch for pytest
        # try:
        _cregion_determine_boundaries_core(cregion, grid)
        # except:
        #     msg = f"error identifying boundaries of cregion {cregion.id}"
        #     pixels = []
        #     if debug_dump:
        #         dump_file = f"dump_cregion_{cregion.id}.py"
        #         cregion_dump(cregion, dump_file, grid.constants)
        #         msg += f"; dumped pixels to file: {dump_file}"
        #     raise Exception(msg)
        # SR_DBG >
    if n_empty > 0:
        log.warning(f"{n_empty}/{cregions.n} regions empty")


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::_cregion_insert_hole_pixel
# :call: v stormtrack::core::cregion::_cregion_insert_shell_pixel
# :call: v stormtrack::core::cregion::_cregion_new_hole
# :call: v stormtrack::core::cregion::_cregion_new_shell
# :call: v stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: v stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: v stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v stormtrack::core::cregion::cpixel_set_region
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregions::cregions_cleanup
cdef void _cregion_determine_boundaries_core(cRegion* cregion, cGrid* grid) except *:
    cdef bint debug = False
    if debug:
        log.debug(f"< _cregion_determine_boundaries_core: {cregion.id}")
    cdef int i
    cdef int j
    cdef int k
    cdef int n_neighbors_feature
    cdef cPixel* cpixel

    if cregion.pixels_n < 1:
        raise Exception("Feature contains no pixels!")

    if cregion.pixels_n == 1:
        cpixel = cregion.pixels[0]
        cpixel_set_region(cpixel, cregion)
        _cregion_insert_shell_pixel(cregion, 0, cpixel)
        return

    # Determine boundary pixels (regardless of which boundary)
    cdef cRegion boundary_pixels = _determine_boundary_pixels_raw(cregion, grid)
    cdef int n_boundary_pixels_raw = boundary_pixels.pixels_n
    if n_boundary_pixels_raw <= 0:
        err = f"no boundary pixels found (n={boundary_pixels.pixels_n})"
        err += f"\nregion {cregion.id} ({cregion.pixels_n} pixels):"
        for i in range(cregion.pixels_max):
            if cregion.pixels[i] is not NULL:
                err += f"\n ({cregion.pixels[i].x}, {cregion.pixels[i].y})"
        raise Exception(err)

    # Group boundary pixels into distinct boundaries (both shells and holes)
    cdef cRegions cboundaries = _reconstruct_boundaries(&boundary_pixels, grid)
    if debug:
        log.debug(f"found {cboundaries.n} boundaries:")

    # Issue error if there are no boundary pixels
    if cboundaries.n == 0:
        err = (
            f"could not reconstruct boundaries of cregion {cregion.id} "
            f"(n={cregion.pixels_n}) from {n_boundary_pixels_raw} boundary pixels"
        )
        for i in range(boundary_pixels.pixels_max):
            if boundary_pixels.pixels[i] is not NULL:
                err += (
                    f"\n {i} ({boundary_pixels.pixels[i].x}, "
                    f"{boundary_pixels.pixels[i].y})"
                )
        raise Exception(err)

    # Clean up ungrouped boundary pixels
    # SR_TODO is reset_connected necessary?
    cregion_cleanup(&boundary_pixels, unlink_pixels=False, reset_connected=True)

    # Categorize boundaries as shells or holes
    cdef bint* boundary_is_shell = categorize_boundaries(&cboundaries, grid)

    # Transfer shells into original cregion
    cdef int i_bnd
    cdef int n_pixels
    cdef int i_pixel
    cdef int j_pixel
    cdef int i_shell = -1
    cdef int i_hole = -1
    cdef cPixel** cpixels_tmp
    cdef cRegion* cboundary
    for i_bnd in range(cboundaries.n):
        cboundary = cboundaries.regions[i_bnd]
        n_pixels = cboundary.pixels_n
        if boundary_is_shell[i_bnd]:
            i_shell += 1
            _cregion_new_shell(cregion)
        else:
            i_hole += 1
            _cregion_new_hole(cregion)

        cpixels_tmp = <cPixel**>malloc(n_pixels*sizeof(cPixel*))

        j_pixel = 0
        for i_pixel in range(cboundary.pixels_max):
            if cboundary.pixels[i_pixel] is not NULL:
                # SR_DBG_PERMANENT <
                if j_pixel >= n_pixels:
                    raise Exception(
                        f"region {cboundary.id}: j_pixel > n_pixels: "
                        f"{j_pixel} >= {n_pixels}"
                    )
                # SR_DBG_PERMANENT >
                cpixels_tmp[j_pixel] = cboundary.pixels[i_pixel]
                j_pixel += 1

        for i_pixel in range(n_pixels):
            cpixel = cpixels_tmp[i_pixel]
            if cpixel is not NULL:
                cpixel_set_region(cpixel, cregion)
                if boundary_is_shell[i_bnd]:
                    _cregion_insert_shell_pixel(cregion, i_shell, cpixel)
                else:
                    _cregion_insert_hole_pixel(cregion, i_hole, cpixel)

        free(cpixels_tmp)

    # SR_ONE_SHELL < TODO remove once it works
    # - # Transfer holes into original cregion
    # - cdef int i_bnd, i_shell, i_hole, i_pixel
    # - i_hole = 0
    # - for i_bnd in range(1, cboundaries.n):
    # -     if is_shell[i_bnd]:
    # -         continue
    # -     cboundary = cboundaries.regions[i_bnd]
    # -     i_hole += 1
    # -     _cregion_new_hole(cregion)
    # -     np = cboundary.pixels_n
    # -     cpixels_tmp = <cPixel**>malloc(np*sizeof(cPixel))
    # -     j = 0
    # -     for i in range(cboundary.pixels_max):
    # -         if cboundary.pixels[i] is not NULL:
    # -             # SR_DBG_PERMANENT <
    # -             if j >= np:
    # -                 err = f"region {cboundary.id}: j={j} >= np={np}"
    # -                 raise Exception(err)
    # -             # SR_DBG_PERMANENT >
    # -             cpixels_tmp[j] = cboundary.pixels[i]
    # -             j += 1
    # -     for i_pixel in range(np):
    # -         cpixel = cpixels_tmp[i_pixel]
    # -         if cpixel is not NULL:
    # -             cpixel_set_region(cpixel, cregion)
    # -             _cregion_insert_hole_pixel(cregion, i_hole, cpixel)
    # -     free(cpixels_tmp)
    # SR_ONE_SHELL >

    # Cleanup
    free(boundary_is_shell)
    cregions_cleanup(&cboundaries, cleanup_regions=True)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::get_matching_neighbor_id
# :call: v stormtrack::core::tables::neighbor_link_stat_table_init
# :call: v stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: v stormtrack::core::cregion::_extract_closed_path
# :call: v stormtrack::core::cregion::_find_link_to_continue
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregion::cregion_insert_pixel
# :call: v stormtrack::core::cregion::cregion_northernmost_pixel
# :call: v stormtrack::core::cregions::cregions_create
# :call: v stormtrack::core::cregions::cregions_link_region
# :call: v stormtrack::core::grid::grid_create_cregion
cdef cRegions _reconstruct_boundaries(cRegion* boundary_pixels, cGrid* grid) except *:
    cdef bint debug = False
    if debug:
        log.debug(
            "\n< _reconstruct_boundaries (boundary_pixels: "
            f"id={boundary_pixels.id}, pixels_n={boundary_pixels.pixels_n})"
        )
    cdef int i
    cdef int j
    cdef int k
    cdef cPixel* cpixel
    cdef cPixel* cpixel_other

    # Define minimal boundary length
    cdef int min_length_boundary = 3

    # SR_TODO Figure out a way to avoid a whole nx*ny array
    # SR_TODO Not really efficient because the features are much smaller!
    neighbor_link_stat_table_init(
        grid.neighbor_link_stat_table, boundary_pixels, &grid.constants,
    )

    # Create containers for boundaries
    cdef cRegion* cregion
    cdef int n_regions = 5
    cdef cRegions boundaries = cregions_create(n_regions)

    #
    # Connect boundaries: distinguish outer boundary from inner boundaries
    #
    # - Start with the northernmost boundary pixel; the corresponding
    #   boundary must be the shell, the others nested holes.
    #
    # - Every boundary pixel holds a list of length connectivity which
    #   contains links to all remaining neighboring boundary points.
    #
    # - All neighbors which either belong to the background or the interior
    #   of the feature (i.e. have not previously been identified as bounday
    #   pixels) are NULL.
    #
    # - Also, all neighboring boundary points which have already been
    #   processed (i.e. all links that have already been followed along
    #   a "boundary chain") are set to NULL as well.
    #
    # - The boundary reconstruction runs until all neighbors are NULL.
    #

    cdef int next_bid = 0
    cdef np.uint8_t i_neighbor
    cdef np.uint8_t i_neighbor_old
    cdef bint done = False
    cdef cPixel* cpixel_start = NULL
    cdef cRegion* current_boundary = NULL
    cdef int iter0
    cdef int iter0_max = 10*boundary_pixels.pixels_n
    cdef int iter1
    cdef int iter1_max = 10*boundary_pixels.pixels_n
    cdef np.int32_t x0
    cdef np.int32_t y0
    cdef np.int32_t x1
    cdef np.int32_t y1
    cdef bint valid_boundary
    for iter0 in range(iter0_max):
        if debug:
            log.debug(f"ITER0 {iter0}/{iter0_max}")

        if iter0 > 0:
            valid_boundary = True

            # If the previous boundary is not closed, discard it
            x0 = current_boundary.pixels[0].x
            y0 = current_boundary.pixels[0].y
            x1 = current_boundary.pixels[current_boundary.pixels_n-1].x
            y1 = current_boundary.pixels[current_boundary.pixels_n-1].y
            if x0 != x1 or y0 != y1:
                if debug:
                    log.debug(f"BOUNDARY NOT CLOSED! ({x0}, {y0}) != ({x1}, {y1})")

                # SR_TODO check if really necessary (once other things fixed)
                # Extract the longest closed segment if there is one
                valid_boundary = _extract_closed_path(current_boundary)

                if debug:
                    log.debug(
                        f"EXTRACTION OF CLOSED PATH {'' if valid_boundary else 'NOT '}"
                        "SUCCESSFUL"
                    )

            # If the previous boundary is too short, discard it
            if current_boundary.pixels_n < min_length_boundary:
                if debug:
                    log.debug(
                        "BOUNDARY TOO SHORT! "
                        f"{current_boundary.pixels_n} < {min_length_boundary}"
                    )
                valid_boundary = False

            if valid_boundary:
                if debug:
                    log.debug("KEEP BOUNDARY")
                cregions_link_region(
                    &boundaries, current_boundary, cleanup=False, unlink_pixels=False,
                )
                current_boundary = NULL
            else:
                if debug:
                    log.debug("DISCARD BOUNDARY")
                cregion_cleanup(
                    current_boundary, unlink_pixels=True, reset_connected=True,
                )

        # Select the pixel and neighbor to start with
        # We're done if there's no unchecked neighbor left
        if iter0 == 0:
            # First iteration: start from the north
            done = True
            cpixel = cregion_northernmost_pixel(boundary_pixels)
            # Select the first existing neighbor
            for i in range(cpixel[0].neighbors_max):
                status = grid.neighbor_link_stat_table[cpixel.x][cpixel.y][i]
                if status > 0:
                    i_neighbor = i
                    done = False
                    break
        else:
            done = _find_link_to_continue(
                &cpixel, &i_neighbor, boundary_pixels, grid.neighbor_link_stat_table,
            )
        if done:
            if debug:
                log.debug("all neighbors of all pixels checked!")
            break
        cpixel_start = cpixel

        # Start new boundary
        if current_boundary is NULL:
            current_boundary = grid_create_cregion(grid)
        cregion_insert_pixel(
            current_boundary, cpixel, link_region=False, unlink_pixel=False,
        )

        # SR_TODO Move loop body into function
        # Loop until boundary closed
        for iter1 in range(iter1_max):
            # DBG_BLOCK <
            if debug:
                log.debug(f"ITER_1({iter1}) ({cpixel.x}, {cpixel.y}) {i_neighbor}")
                log.debug(f"boundaries: {boundaries.n}")
                log.debug(f"current boundary: {current_boundary.pixels_n} pixels:")
                # for i in range(current_boundary.pixels_max):
                #    if current_boundary.pixels[i] is not NULL:
                #        log.debug(
                #            f"[{i}] ({current_boundary.pixels[i].x}, "
                #            f"{current_boundary.pixels[i].y})"
                #        )
            # DBG_BLOCK >

            # Check if the current boundary is finished (full-circle)
            if iter1 > 0 and cpixel.id == cpixel_start.id:
                done = True
                for i in range(1, cpixel.neighbors_max):
                    with cython.cdivision(True):
                        j = (
                            i + get_matching_neighbor_id(
                                i_neighbor_old, cpixel.neighbors_max
                            )
                        ) % cpixel.neighbors_max
                    if grid.neighbor_link_stat_table[cpixel.x][cpixel.y][j] > 0:
                        done = False
                        break
                    if grid.neighbor_link_stat_table[cpixel.x][cpixel.y][j] == 0:
                        break
                # if neighbor_link_stat_table[cpixel.x][cpixel.y][i_neighbor] < 2:
                if done:
                    if debug:
                        log.debug(" *** BOUNDARY DONE [0] ***")
                    break

            # Advance to neighbor
            grid.neighbor_link_stat_table[cpixel.x][cpixel.y][i_neighbor] = 0
            if debug:
                log.debug(
                    f"neighbor_link_stat_table[{cpixel.x}][{cpixel.y}][{i_neighbor}] = 0"
                )
            cpixel = cpixel.neighbors[i_neighbor]
            i_neighbor_old = i_neighbor
            i_neighbor = get_matching_neighbor_id(i_neighbor, cpixel.neighbors_max)

            if debug:
                log.debug("ADVANCE: ({cpixel.x}, {cpixel.y}) (from {i_neighbor})")

            if grid.neighbor_link_stat_table[cpixel.x][cpixel.y][i_neighbor] > 1:
                grid.neighbor_link_stat_table[cpixel.x][cpixel.y][i_neighbor] = 1
            i_neighbor += 1

            # Add the pixel to the current boundary
            cregion_insert_pixel(
                current_boundary, cpixel, link_region=False, unlink_pixel=False,
            )

            # Get next neighbor
            if debug:
                log.debug(f"find neighbor of ({cpixel.x}, {cpixel.y})")
            for i in range(i_neighbor, i_neighbor + cpixel.neighbors_max):
                # SR_TODO add cython.cdivision directive to modulo
                with cython.cdivision(True):
                    i_neighbor = i%cpixel.neighbors_max
                if cpixel.neighbors[i_neighbor] is NULL:
                    if debug:
                        log.debug(f" -> ({i_neighbor}) NULL")
                    continue
                if debug:
                    log.debug(
                        f" -> ({i_neighbor}) "
                        f"({cpixel.neighbors[i_neighbor].x}, {cpixel.neighbors[i_neighbor].y}) "
                        f"{grid.neighbor_link_stat_table[cpixel.x][cpixel.y][i_neighbor]}"
                    )
                if grid.neighbor_link_stat_table[cpixel.x][cpixel.y][i_neighbor] > 0:
                    break
            else:
                if debug:
                    log.debug(" *** BOUNDARY DONE [1] ***")
                break
            if debug:
                log.debug(f" => next neighbor of ({cpixel.x}, {cpixel.y}): {i_neighbor}")
        else:
            raise Exception(f"identify_regions: loop(1) timed out (nmax={iter1_max})")

        # DBG_BLOCK <
        if debug:
            log.debug(f"finished boundary: {current_boundary.pixels_n} pixels:")
            for i in range(current_boundary.pixels_max):
                if current_boundary.pixels[i] is NULL:
                    continue
                log.debug(
                    f"[{i}] ({current_boundary.pixels[i].x}, "
                    f"{current_boundary.pixels[i].y})"
                )
            log.debug("\n================================\n")
        # DBG_BLOCK >
    else:
        raise Exception(f"identify_regions: loop # 0 timed out (nmax={iter0_max})")

    # Reset neighbor link status table
    neighbor_link_stat_table_reset(grid.neighbor_link_stat_table, &grid.constants)

    if debug:
        log.debug("> _reconstruct_boundaries")
    return boundaries


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_check_validity
# :call: v stormtrack::core::cregion_boundaries::boundary_must_be_a_shell
# :call: v stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: v stormtrack::core::cregion_boundaries::cregions_find_northernmost_uncategorized_region
# :call: v stormtrack::core::cpixel::cpixel_angle_to_neighbor
# :call: v stormtrack::core::cregion::sign
cdef bint* categorize_boundaries(cRegions* boundaries, cGrid* grid) except *:
    cdef bint debug = False
    # cdef bint debug = True  # SR_DBG
    if debug:
        log.debug(f"< categorize_boundaries: {boundaries.n}")
    cdef int ib
    cdef int ib_sel
    cdef int ic
    cdef int dx
    cdef int dy
    cdef int d_angle
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
    cdef int i_angle
    cdef int n_angle
    cdef int i_d_angle
    cdef int n_d_angle
    cdef int i_px_idcs
    cdef int n_px_idcs
    cdef int* px_idcs = NULL
    cdef int angle_pre
    cdef int n_pixels_eff
    cdef int iter_i
    cdef int iter_max=10000
    for iter_i in range(iter_max):
        if debug:
            log.debug(f"  ----- ITER {iter_i} ({iter_i + 1:,}/{iter_max:,}) -----")

        # Select unprocessed boundary with northernost pixel
        ib_sel = cregions_find_northernmost_uncategorized_region(boundaries, categorized)
        if debug:
            log.debug(f"northernmost uncategorized region: {ib_sel}")
        if ib_sel < 0:
            # All boundaries categorized!
            if debug:
                log.debug(f"  ----- DONE {iter_i} ({iter_i + 1}/{iter_max}) -----")
            break
        boundary = boundaries.regions[ib_sel]
        cregion_check_validity(boundary, ib_sel)
        n_pixels = boundary.pixels_n
        if debug:
            log.debug(f"  process boundary {ib_sel} ({ib_sel + 1}/{n_bnds}) ({n_pixels} px)")

        # SR_DBG_NOW <
        # - print(f"\n boundary # {ib_sel} ({n_pixels}):")
        # - for i in range(n_pixels):
        # -     print(f" {i:2} ({boundary.pixels[i].x:2},{boundary.pixels[i].y:2})")
        # SR_DBG_NOW >

        d_angles = <int*>malloc(n_pixels*sizeof(int))
        angles   = <int*>malloc(n_pixels*sizeof(int))
        px_idcs  = <int*>malloc(n_pixels*sizeof(int))
        for i in range(n_pixels):
            d_angles[i] = 999
            angles  [i] = 999
            px_idcs [i] = 999
        i_angle = -1
        n_angle = 0
        i_d_angle = -1
        n_d_angle = 0
        i_px_idcs = -1
        n_px_idcs = 0

        if debug: print(f"<<<<< ({n_pixels})") # SR_DBG_NOW

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

            i_px_idcs += 1
            n_px_idcs += 1
            px_idcs[i_px_idcs] = i_px
            cpixel = boundary.pixels[i_px]
            # - print('# ', i_px, px_idcs[i_angle], (i_px_idcs, n_px_idcs), (i_angle, n_angle), (i_d_angle, n_d_angle), (cpixel.x, cpixel.y)) # SR_DBG_NOW

            # Determine angle from previous to current pixel
            angle = cpixel_angle_to_neighbor(cpixel, cpixel_pre)
            i_angle += 1
            n_angle += 1
            angles[i_angle] = angle
            px_idcs[i_angle] = i_px
            if debug: print(f"    ({cpixel_pre.x}, {cpixel_pre.y}) -> ({cpixel.x}, {cpixel.y}): {angle}")  # SR_DBG_NOW

            if angle_pre != -999:
                d_angle = angle - angle_pre
                if abs(d_angle) > 180:
                    # Limit to +-180
                    d_angle = d_angle - sign(d_angle) * 360
                i_d_angle += 1
                n_d_angle += 1
                d_angles[i_d_angle] = d_angle

                if d_angle == 180:
                    # Change to opposite direction indicates an isolated
                    # boundary pixel, which can simply be ignored
                    # # SR_TMP <
                    # if i_px_idcs < 2:
                    #     raise NotImplementedError(f"i_px_idcs: {i_px_idcs} < 2")
                    # # SR_TMP >
                    cpixel_pre2 = boundary.pixels[px_idcs[i_px_idcs - 2]]

                    # print(
                    #     f"! {i_px_idcs - 2:2}/{px_idcs[i_px_idcs - 2]:2}"
                    #     f"({cpixel_pre2.x:2},{cpixel_pre2.y:2})"
                    #     f" {i_px_idcs - 1:2}/{px_idcs[i_px_idcs - 1]:2}"
                    #     f"({cpixel_pre1.x:2},{cpixel_pre.y:2})"
                    #     f" {i_px_idcs - 0:2}/{px_idcs[i_px_idcs - 0]:2}"
                    #     f"({cpixel.x:2},{cpixel.y:2})"
                    # )
                    # raise Exception("ambiguous angle: 180 or -180?")

                    # # SR_DBG <
                    # if cpixel_pre2.x != cpixel.x:
                    #     raise Exception("x_pre2 != x", cpixel_pre2.x, cpixel.x)
                    # if cpixel_pre2.y != cpixel.y:
                    #     raise Exception("y_pre2 != y", cpixel_pre2.y, cpixel.y)
                    # # SR_DBG >

                    # SR_DBG_NOW <
                    # - print()
                    # - for i in range(n_angle): print(i, i_angle, n_angle, angles[i])
                    # - print()
                    # - for i in range(n_d_angle): print(i, i_d_angle, n_d_angle, d_angles[i])
                    # - print()
                    # - for i in range(n_px_idcs): print(i, i_px_idcs, n_px_idcs, px_idcs[i])
                    # - print()
                    # SR_DBG_NOW >

                    i_px_idcs  = max(i_px_idcs  - 2, -1)
                    n_px_idcs  = max(n_px_idcs  - 2,  0)
                    i_d_angle = max(i_d_angle - 2, -1)
                    n_d_angle = max(n_d_angle - 2,  0)
                    i_angle  = max(i_angle  - 2, -1)
                    n_angle  = max(n_angle  - 2,  0)
                    n_pixels_eff -= 2
                    angle_pre = angles[i_angle]
                    cpixel_pre = cpixel

                    # SR_DBG_NOW <
                    # - print()
                    # - for i in range(n_angle): print(i, i_angle, n_angle, angles[i])
                    # - print()
                    # - for i in range(n_d_angle): print(i, i_d_angle, n_d_angle, d_angles[i])
                    # - print()
                    # - for i in range(n_px_idcs): print(i, i_px_idcs, n_px_idcs, px_idcs[i])
                    # - print()
                    # - print((i_px_idcs, n_px_idcs), (i_d_angle, n_d_angle), (i_angle, n_angle), n_pixels_eff, angle_pre)
                    # - print(" OK cpixel_pre2 == cpixel")
                    # SR_DBG_NOW >
                    continue
                # SR_TMP >

                # DBG_BLOCK <
                if debug:
                    log.debug(
                        f"  {ic:2} ({cpixel_pre.x:2}, {cpixel_pre.y:2}) "
                        f"-> ({cpixel.x:2}, {cpixel.y:2}) {angle_pre:4} "
                        f"-> {angle:4} : {d_angle:4}"
                    )
                # DBG_BLOCK >

            cpixel_pre = cpixel
            angle_pre = angle

        if not categorized[ib_sel]:
            categorize_boundary_is_shell(ib_sel, d_angles, n_d_angle, boundary_is_shell)
            categorized[ib_sel] = True

        free(angles)
        free(d_angles)
        free(px_idcs)
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
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
cdef bint boundary_must_be_a_shell(int n_pixels_eff, cGrid* grid) except -1:
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
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v --- calling ---
cdef void categorize_boundary_is_shell(
    int ib_sel, int* d_angles, int n_d_angle, bint* boundary_is_shell,
) except *:
    # Sum up the angles
    cdef int delta_angle_tot = 0
    cdef int i_d_angle
    for i_d_angle in range(n_d_angle):
        # - print(f" {i_d_angle:2} {d_angles[i_d_angle]:4} {int(delta_angle_tot):4}")
        delta_angle_tot += d_angles[i_d_angle]
    # print(f"       {int(delta_angle_tot):4}")

    # Categorize the boundary
    if delta_angle_tot == -360:
        boundary_is_shell[ib_sel] = True
        # print(">"*5+" SHELL")
    elif delta_angle_tot == 360:
        boundary_is_shell[ib_sel] = False
        # print(">"*5+" SHELL")
    else:
        raise Exception(
            f"categorization of boundary # {ib_sel} failed: "
            f"total angle {delta_angle_tot} != +-360"
        )


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_northernmost_pixel
cdef int cregions_find_northernmost_uncategorized_region(
    cRegions* boundaries, bint* categorized,
) except -999:
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
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v --- calling ---
cdef inline int sign(int num):
    if num >= 0:
        return 1
    else:
        return -1
