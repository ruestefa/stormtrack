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
    cdef bint* boundaries_is_shell = categorize_boundaries(&cboundaries, grid)

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
        if boundaries_is_shell[i_bnd]:
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
                if boundaries_is_shell[i_bnd]:
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
    free(boundaries_is_shell)
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
# :call: v stormtrack::core::cregion::cregion_check_validity
# :call: v stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: v stormtrack::core::cregion_boundaries::cregions_find_northernmost_uncategorized_region
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
cdef bint* categorize_boundaries(cRegions* boundaries, cGrid* grid) except *:
    """Categorize boundaries as shell or hole.
    
    Algorithm:
    
    - Find the uncategorized boundary with the northernmost pixel.
      It must be a shell, because all other boundaries are either contained
      by it (holes or nested shells), or other shells located further south.
    
    - Mark all pixels inside this shell, and collect all uncategorized
      boundaries comprised of such pixels (one match is sufficient, but if
      one pixel matches, then all should; check that in debug mode).
    
    - Among these boundaries, select the northernmost, which (by the same
      logic as before) must be a hole. Then mark all contained pixels and
      select all nested boundaries. Repeat until there are no more nested
      boundaries.
    
    - Repeat until all boundaries have been assigned.
    
    """
    cdef bint debug = False
    if debug:
        log.debug(f"< categorize_boundaries: {boundaries.n}")
    cdef bint* boundaries_is_shell = <bint*>malloc(boundaries.n * sizeof(bint))
    cdef bint* boundaries_is_categorized = <bint*>malloc(boundaries.n * sizeof(bint))
    cdef int i_bnd
    for i_bnd in range(boundaries.n):
        boundaries_is_categorized[i_bnd] = False
    cdef cRegion* boundary = NULL
    cdef int iter_i
    cdef int iter_max=10000
    for iter_i in range(iter_max):
        if debug:
            log.debug(f"  ----- ITER {iter_i} ({iter_i + 1:,}/{iter_max:,}) -----")
        i_bnd_sel = cregions_find_northernmost_uncategorized_region(
            boundaries, boundaries_is_categorized
        )
        if i_bnd_sel < 0:
            break
        boundary = boundaries.regions[i_bnd_sel]
        cregion_check_validity(boundary, i_bnd_sel)
        if debug:
            log.debug(
                f"  process boundary {i_bnd_sel} ({i_bnd_sel + 1}/{boundaries.n})"
                f" ({boundary.pixels_n} px)"
            )
        boundaries_is_shell[i_bnd_sel] = categorize_boundary_is_shell(grid, boundary)
        boundaries_is_categorized[i_bnd_sel] = True
    else:
        raise Exception(f"timeout after {iter_max:,} iterations")
    if debug:
        log.debug(f"  ----- DONE {iter_i} ({iter_i + 1}/{iter_max}) -----")
    free(boundaries_is_categorized)
    if debug:
        log.debug("> categorize_boundaries: done")
    return boundaries_is_shell


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_advance
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_curr
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_init
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_new
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_next
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_prev
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_reset
# :call: v stormtrack::core::cregion_boundaries::cpixel_angle_to_neighbor
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef bint categorize_boundary_is_shell(cGrid *grid, cRegion *boundary) except -1:
    cdef bint debug = False
    cdef int i_px
    cdef int i
    cdef int n_px = boundary.pixels_n
    cdef bint *skipped = <bint*>malloc(n_px * sizeof(bint))
    for i in range(n_px):
        skipped[i] = False
    cdef cPixel *px_prev
    cdef cPixel *px_curr
    cdef cPixel *px_next
    cdef int angle
    cdef int angle_sum = 0
    cdef cPixelCycle _pxs = cPixelCycle_new(debug=debug)
    cdef cPixelCycle *pxs = &_pxs
    cPixelCycle_init(pxs, boundary)
    if pxs.n_eff == 0:
        # No effective pixels: Must be a shell!
        cPixelCycle_reset(pxs)
        return True
    for i in range(pxs.n_eff):
        px_prev = cPixelCycle_prev(pxs)
        px_curr = cPixelCycle_curr(pxs)
        px_next = cPixelCycle_next(pxs)
        angle = cpixel_angle_to_neighbor(px_prev, px_curr, px_next)
        angle_sum += angle
        if debug:
            print(
                f"[{i}] "
                f"({px_prev.x}, {px_prev.y}) -> "
                f"({px_curr.x}, {px_curr.y}) -> "
                f"({px_next.x}, {px_next.y})"
                f" : {angle} ({angle_sum})"
            )
        cPixelCycle_advance(pxs)
    if angle_sum == -360:
        return True
    elif angle_sum == 360:
        return False
    else:
        raise Exception("angle sum not +-360", angle_sum)
    cPixelCycle_reset(pxs)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::PixelCycle__advance_to_unskipped
# :call: > stormtrack::core::cregion_boundaries::PixelCycle__i_next
# :call: > stormtrack::core::cregion_boundaries::PixelCycle__i_prev
# :call: > stormtrack::core::cregion_boundaries::PixelCycle__init_skip
# :call: > stormtrack::core::cregion_boundaries::PixelCycle_advance
# :call: > stormtrack::core::cregion_boundaries::PixelCycle_curr
# :call: > stormtrack::core::cregion_boundaries::PixelCycle_new
# :call: > stormtrack::core::cregion_boundaries::PixelCycle_next
# :call: > stormtrack::core::cregion_boundaries::PixelCycle_prev
# :call: > stormtrack::core::cregion_boundaries::PixelCycle_reset
# :call: > stormtrack::core::cregion_boundaries::ategorize_boundary_is_shell
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
cdef struct cPixelCycle:
    cRegion *_pixels
    bint *_skip
    int i
    int n
    int n_eff
    bint debug


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::PixelCycle_init
# :call: > stormtrack::core::cregion_boundaries::PixelCycle_new
# :call: > stormtrack::core::cregion_boundaries::ategorize_boundary_is_shell
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_reset
cdef cPixelCycle cPixelCycle_new(bint debug=False) except *:
    """Create a new instance of ``cPixelCycle``."""
    self.debug = debug
    if self.debug:
        print("cPixelCycle: create new object")
    cdef cPixelCycle self
    self._pixels = NULL
    self._skip = NULL
    cPixelCycle_reset(&self)
    return self


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle_init
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
cdef void cPixelCycle_reset(cPixelCycle *self) except *:
    if self.debug:
        print("cPixelCycle: reset")
    if self._pixels is not NULL:
        self._pixels = NULL
    if self._skip is not NULL:
        free(self._skip)
        self._skip = NULL
    self.i = -999
    self.n = -999
    self.n_eff = -999


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle__init_skip
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle_reset
# :call: v stormtrack::core::structs::cRegion
cdef void cPixelCycle_init(cPixelCycle *self, cRegion *pixels, int i=0) except *:
    """Initialize a new instance of ``cPixelCycle``."""
    cPixelCycle_reset(self)
    cdef int i_px
    cdef cPixel *px
    if self.debug:
        print(f"cPixelCycle: initialize with {pixels.pixels_n} pixels:")
        for i_px in range(pixels.pixels_n):
            px = pixels.pixels[i_px]
            print(f"  ({px.x}, {px.y})")
    self._pixels = pixels
    self.i = i
    self.n = pixels.pixels_n
    self._skip = <bint*>malloc(self.n * sizeof(bint))
    cPixelCycle__init_skip(self)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle_init
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::PixelCycle
# :call: v stormtrack::core::cregion_boundaries::PixelCycle__advance_to_unskipped
# :call: v stormtrack::core::cregion_boundaries::PixelCycle__i_next
# :call: v stormtrack::core::cregion_boundaries::PixelCycle_curr
# :call: v stormtrack::core::cregion_boundaries::PixelCycle_next
# :call: v stormtrack::core::cregion_boundaries::PixelCycle_prev
# :call: v stormtrack::core::structs::cPixel
cdef void cPixelCycle__init_skip(cPixelCycle *self) except *:
    """Skip redundant and isolated pixels that are categorization-irrelevant."""
    if self.debug:
        print(f"cPixelCycle: identify pixels to be skipped (tot: {self.n})")
    cdef int i
    cdef cPixel *px
    for i in range(self.n):
        self._skip[i] = False

    # Skip redundant start/end pixel
    if cpixel_equals(self._pixels.pixels[0], self._pixels.pixels[self.n - 1]):
        self._skip[self.n - 1] = True
        if self.debug:
            px = self._pixels.pixels[self.n - 1]
            print(f"  skip [{self.n - 1}] ({px.x}, {px.y}) (duplicate end)")

    # Skip all isolated pixels, i.e., with only one neighbor (prev equals next)
    cdef int i_bak
    cdef int i_prev
    cdef bint *skip_tmp = <bint*>malloc(self.n * sizeof(bint))
    cdef int n_eff_bak = self.n_eff
    cdef int iter_max = 99
    cdef int iter_i
    for iter_i in range(iter_max):
        for i in range(self.n):
            skip_tmp[i] = self._skip[i]
        i_bak = self.i
        for i in range(self.n):
            j = i_bak + i
            if j >= self.n:
                j -= self.n
            self.i = j
            if cpixel_equals(cPixelCycle_prev(self), cPixelCycle_next(self)):
                if self.debug:
                    px = cPixelCycle_curr(self)
                    print(f"  skip [{j}] ({px.x}, {px.y}) (isolated)")
                skip_tmp[j] = True
                skip_tmp[cPixelCycle__i_next(self)] = True
        self.i = i_bak
        self.n_eff = self.n
        for i in range(self.n):
            self._skip[i] = skip_tmp[i]
            if self._skip[i]:
                self.n_eff -= 1
        if self.n_eff == 0:
            # Only isolated pixels, e.g., a line: Must be a shell!
            return
        if self.debug:
            print(f"  effective no. pixels: {self.n_eff}/{self.n}")
        if self.n_eff == n_eff_bak:
            break
        n_eff_bak = self.n_eff
    else:
        raise Exception("timeout after {iter_i} iterations")
    free(skip_tmp)
    cPixelCycle__advance_to_unskipped(self)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__init_skip
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle_advance
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
cdef void cPixelCycle__advance_to_unskipped(cPixelCycle *self, int min_step=0) except *:
    """Advance to the next unskipped pixel, if necessary."""
    for i in range(self.n):
        j = self.i + i
        if j >= self.n:
            j -= self.n
        if i >= min_step and not self._skip[j]:
            self.i = j
            break
    else:
        raise Exception("no unskipped pixels")


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle__advance_to_unskipped
cdef void cPixelCycle_advance(cPixelCycle *self) except *:
    """Advance by one pixel."""
    if self.debug:
        print(f"cPixelCycle: advance from {self.i}")
    cPixelCycle__advance_to_unskipped(self, min_step=1)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__init_skip
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
cdef cPixel *cPixelCycle_curr(cPixelCycle *self) except *:
    """Get the current pixel."""
    if self.debug:
        print("cPixelCycle: get curr: ", end="")
    cdef int i = self.i
    if self._skip[i]:
        raise Exception("current pixel cannot be skipped", self.i)
    if self.debug:
        print(f"[{i}] ", end="")
    cdef cPixel *px = self._pixels.pixels[i]
    if self.debug:
        print(f"({px.x}, {px.y})")
    return px


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__init_skip
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle__i_prev
cdef cPixel *cPixelCycle_prev(cPixelCycle *self) except *:
    """Get the previous pixel."""
    if self.debug:
        print(f"cPixelCycle: get prev: ", end="")
    cdef int i = cPixelCycle__i_prev(self)
    if self.debug:
        print(f"[{i}] ", end="")
    cdef cPixel *px = self._pixels.pixels[i]
    if self.debug:
        print(f"({px.x}, {px.y})")
    return px


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle_prev
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
cdef int cPixelCycle__i_prev(cPixelCycle *self) except -1:
    """Get the index of the previous pixel."""
    cdef int i
    cdef int j
    for j in range(self.n):
        i = self.i - j - 1
        if i < 0:
            i += self.n
        if not self._skip[i]:
            break
    else:
        raise Exception("no prev pixel found")
    return i


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__init_skip
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle__i_next
cdef cPixel *cPixelCycle_next(cPixelCycle *self) except *:
    """Get the next pixel."""
    if self.debug:
        print(f"cPixelCycle: get next: ", end="")
    cdef int i = cPixelCycle__i_next(self)
    if self.debug:
        print(f"[{i}] ", end="")
    cdef cPixel *px = self._pixels.pixels[i]
    if self.debug:
        print(f"({px.x}, {px.y})")
    return px


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__init_skip
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle_next
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::cPixelCycle
cdef int cPixelCycle__i_next(cPixelCycle *self) except -1:
    """Get the index of the next pixel."""
    cdef int i
    cdef int j
    cdef int n = self._pixels.pixels_n
    for j in range(n):
        i = self.i + j + 1
        if i >= n:
            i -= n
        if not self._skip[i]:
            break
    else:
        raise Exception("no next pixel found")
    return i


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: v --- calling ---
# :call: v stormtrack::core::cregion_boundaries::_direction_between_pixels
# :call: v stormtrack::core::structs::cPixel
cdef int cpixel_angle_to_neighbor(
    cPixel* cpixel0, cPixel* cpixel1, cPixel* cpixel2,
) except -1:
    cdef bint debug = False
    cdef int dir1 = _direction_between_pixels(cpixel0, cpixel1)
    cdef int dir2 = _direction_between_pixels(cpixel1, cpixel2)
    cdef int angle
    angle = dir2 - dir1
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    if debug:
        print(
            f"({cpixel0.x}, {cpixel0.y})-{dir1}->({cpixel1.x}, {cpixel1.y})"
            f"-{dir2}->({cpixel2.x}, {cpixel2.y}): {angle}"
        )
    return angle


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::cpixel_angle_to_neighbor
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
cdef int _direction_between_pixels(cPixel* cpixel0, cPixel* cpixel1):
    cdef dx = cpixel1.x - cpixel0.x
    cdef dy = cpixel1.y - cpixel0.y
    cdef int dir
    if dx == 1 and dy == 0:
        dir = 0
    elif dx == 1 and dy == 1:
        dir = 45
    elif dx == 0 and dy == 1:
        dir = 90
    elif dx == -1 and dy == 1:
        dir = 135
    elif dx == -1 and dy == 0:
        dir = 180
    elif dx == -1 and dy == -1:
        dir = 225
    elif dx == 0 and dy == -1:
        dir = 270
    elif dx == 1 and dy == -1:
        dir = 315
    else:
        raise Exception(
            f"cannot direction between pixels",
            (cpixel0.x, cpixel0.y),
            (cpixel1.x, cpixel1.y),
        )
    return dir


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::cregion::cregion_northernmost_pixel
cdef int cregions_find_northernmost_uncategorized_region(
    cRegions* boundaries, bint* boundaries_is_categorized,
) except -999:
    cdef cPixel* cpixel = NULL
    cdef cPixel* cpixel_sel = NULL
    cdef int i_bnd_sel = -1
    cdef int i_bnd
    for i_bnd in range(boundaries.n):
        if not boundaries_is_categorized[i_bnd]:
            cpixel = cregion_northernmost_pixel(boundaries.regions[i_bnd])
            if (
                cpixel_sel is NULL
                or cpixel.y > cpixel_sel.y
                or cpixel.y == cpixel_sel.y and cpixel.x < cpixel_sel.x
            ):
                cpixel_sel = cpixel
                i_bnd_sel = i_bnd
    return i_bnd_sel
