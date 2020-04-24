# !/usr/bin/env python3

from __future__ import print_function

# C: C libraries
from libc.math cimport pow
from libc.math cimport sqrt
from libc.stdlib cimport exit
from libc.stdlib cimport free
from libc.stdlib cimport malloc

# C: Third-party
cimport cython
cimport numpy as np
from cython.parallel cimport prange

# Standard library
import logging as log
import os
import sys

# Third-party
import cython
import numpy as np


# pixel_done_table_alloc
# pixel_done_table_init
# pixel_done_table_reset
# pixel_done_table_cleanup
# pixel_region_table_alloc
# pixel_region_table_alloc_grid
# pixel_region_table_alloc_pixels
# pixel_region_table_insert_region
# cregion_rank_slots_insert_region
# cregion_rank_slots_extend
# pixel_region_table_alloc_pixel
# pixel_region_table_init_regions
# pixel_region_table_grow
# pixel_region_table_cleanup_pixels
# pixel_region_table_reset
# pixel_region_table_reset_region
# pixel_region_table_reset_regions
# pixel_region_table_reset_slots
# cregion_rank_slots_reset
# cregion_rank_slots_copy
# _pixel_region_table_cleanup_entry
# pixel_region_table_cleanup
# pixel_status_table_init_feature
# pixel_status_table_reset_feature
# pixel_status_table_alloc
# pixel_status_table_reset
# pixel_status_table_cleanup
# neighbor_link_stat_table_alloc
# neighbor_link_stat_table_reset
# neighbor_link_stat_table_reset_pixels
# neighbor_link_stat_table_cleanup
# neighbor_link_stat_table_init
# get_direct_neighbor_index


# :call: > --- CALLERS ---
# :call: > core::identification::csplit_regiongrow_levels
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelDoneTable
# :call: v core::core::structs::cConstants
cdef void pixel_done_table_alloc(PixelDoneTable* table, cConstants* constants):
    cdef int i
    cdef int j
    if table[0] is not NULL:
        log.error("pixel_done_table_alloc: table must be NULL")
        exit(4)
    table[0] = <bint**>malloc(constants.nx*sizeof(bint*))
    for i in range(constants.nx):
        table[0][i] = <bint*>malloc(constants.ny*sizeof(bint))
        for j in range(constants.ny):
            table[0][i][j] = True


# :call: > --- CALLERS ---
# :call: > core::identification::extract_subregions_level
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelDoneTable
# :call: v core::core::structs::cPixel
# :call: v core::core::structs::cRegion
cdef void pixel_done_table_init(
    PixelDoneTable table, cRegion* cregion, np.float32_t level,
):
    cdef int i_pixel
    cdef cPixel* cpixel
    for i_pixel in prange(cregion.pixels_max, nogil=True):
        cpixel = cregion.pixels[i_pixel]
        if cpixel is not NULL:
            if cregion.pixels[i_pixel].v >= level:
                table[cpixel.x][cpixel.y] = False


# :call: > --- CALLERS ---
# :call: > core::identification::extract_subregions_level
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelDoneTable
# :call: v core::core::structs::cPixel
# :call: v core::core::structs::cRegion
cdef void pixel_done_table_reset(PixelDoneTable table, cRegion* cregion):
    cdef int i_pixel
    cdef cPixel* cpixel
    for i_pixel in prange(cregion.pixels_max, nogil=True):
        cpixel = cregion.pixels[i_pixel]
        if cpixel is not NULL:
            table[cpixel.x][cpixel.y] = True


# :call: > --- CALLERS ---
# :call: > core::typedefs::grid_cleanup
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelDoneTable
# :call: v core::core::structs::cConstants
cdef void pixel_done_table_cleanup(PixelDoneTable table, cConstants* constants):
    cdef int i
    for i in prange(constants.nx, nogil=True):
        free(table[i])
    free(table)


# :call: > --- CALLERS ---
# :call: > core::typedefs::Grid::__cinit__
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cConstants
# :call: v core::core::structs::cRegionRankSlot
# :call: v core::core::structs::cRegionRankSlots
cdef void pixel_region_table_alloc(
    PixelRegionTable* table, int n_slots, cConstants* constants,
):
    """Table to store links to regions for each pixel, plus the neighbor rank.

    For each pixel there's an n_slots*2 array, i.e. each pixel can belong to
    a maximum of n_slots regions. Each entry consists of two integers: the
    region index and the neighbor rank.

    The region index is the index of the region in the cRegions.regions array.
    Note that this is NOT the region ID!

    The neighbor rank indicates whether the pixel is a direct (0) or only an
    indirect neighbor (1), to any pixel of the boundary we're growing from,
    i.e. whether it shares a face or is only connected diagnoally.

    Note that 4-connectivity by definition only allows for direct neighbors.

    For seed pixels the respective region (again NOT the region ID) is stored
    as the [0][0] element. (Here the region ID would actually make more sense,
    just to be more robust, but for the sake of consistency the index in the
    seeds cRegion.regions array is chosen.)
    """
    # print(f"< pixel_region_table_alloc {constants.nx}x{constants.ny}x{n_slots}")
    cdef int i
    cdef int j
    cdef int k
    if table[0] is not NULL:
        log.error("pixel_region_table_alloc: table must be NULL")
        exit(4)
    table[0] = <cRegionRankSlots**>malloc(constants.nx*sizeof(cRegionRankSlots*))
    for i in prange(constants.nx, nogil=True):
        table[0][i] = <cRegionRankSlots*>malloc(constants.ny*sizeof(cRegionRankSlots))
        for j in range(constants.ny):
            table[0][i][j].slots = <cRegionRankSlot*>malloc(
                    n_slots*sizeof(cRegionRankSlot))
            table[0][i][j].max = n_slots
            table[0][i][j].n = 0
            for k in range(n_slots):
                table[0][i][j].slots[k].region = NULL
                table[0][i][j].slots[k].rank = -1
    # print("> pixel_region_table_alloc")


# :call: > --- CALLERS ---
# :call: > core::identification::_find_features_threshold_random_seeds
# :call: > core::identification::c_find_features_2d_threshold_seeds
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::identification::feature_split_regiongrow
# :call: > core::identification::features_find_neighbors
# :call: > core::identification::find_features_2d_threshold
# :call: > core::identification::merge_adjacent_features
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cConstants
# :call: v core::core::structs::cRegionRankSlots
cdef void pixel_region_table_alloc_grid(
    PixelRegionTable* table, cConstants* constants,
) except *:
    # print(f"< pixel_region_table_alloc_grid {constants.nx}x{constants.ny}")
    cdef int i
    cdef int j
    if table[0] is not NULL:
        err = "pixel_region_table_alloc_grid: table must be NULL"
        raise Exception(err)
    table[0] = <cRegionRankSlots**>malloc(constants.nx*sizeof(cRegionRankSlots*))
    for i in prange(constants.nx, nogil=True):
        table[0][i] = <cRegionRankSlots*>malloc(constants.ny*sizeof(cRegionRankSlots))
        for j in range(constants.ny):
            table[0][i][j].slots = NULL
            table[0][i][j].max = 0
            table[0][i][j].n = 0
    # print("> pixel_region_table_alloc_grid")


# :call: > --- CALLERS ---
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::csplit_regiongrow_core
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::tables::pixel_region_table_grow
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cRegion
# :call: v core::core::tables::pixel_region_table_alloc_pixel
cdef void pixel_region_table_alloc_pixels(
    PixelRegionTable table, int n_slots, cRegion* cregion,
):
    # print(f"< pixel_region_table_alloc_pixels {n_slots}")
    cdef int i_pixel
    cdef int i_slot
    cdef np.int32_t x
    cdef np.int32_t y
    for i_pixel in prange(cregion.pixels_istart, cregion.pixels_iend, nogil=True):
        if cregion.pixels[i_pixel] is not NULL:
            pixel_region_table_alloc_pixel(
                table, cregion.pixels[i_pixel].x, cregion.pixels[i_pixel].y, n_slots,
            )


# :call: > --- CALLERS ---
# :call: > core::identification::determine_shared_boundary_pixels
# :call: > core::identification::regiongrow_resolve_multi_assignments
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cRegion
# :call: v core::core::tables::cregion_rank_slots_insert_region
cdef void pixel_region_table_insert_region(
    PixelRegionTable table,
    np.int32_t x,
    np.int32_t y,
    cRegion* cregion,
    np.int8_t rank,
):
    cregion_rank_slots_insert_region(
            &table[x][y],
            cregion,
            rank,
        )


# :call: > --- CALLERS ---
# :call: > core::identification::resolve_multi_assignment_best_connected_region
# :call: > core::identification::resolve_multi_assignment_biggest_region
# :call: > core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > core::identification::resolve_multi_assignment_strongest_region
# :call: > core::tables::pixel_region_table_insert_region
# :call: v --- CALLING ---
# :call: v core::core::structs::cRegion
# :call: v core::core::structs::cRegionRankSlots
# :call: v core::core::tables::cregion_rank_slots_extend
cdef void cregion_rank_slots_insert_region(
    cRegionRankSlots* slots, cRegion* cregion, np.int8_t rank,
):
    cdef int i
    cdef int n = slots.n
    if n == slots.max:
        cregion_rank_slots_extend(slots)
    for i in range(slots.n):
        if slots.slots[i].region.id == cregion.id:
            return
    slots.slots[slots.n].region = cregion
    slots.slots[slots.n].rank = rank
    slots.n += 1


# :call: > --- CALLERS ---
# :call: > core::tables::cregion_rank_slots_insert_region
# :call: v --- CALLING ---
# :call: v core::core::structs::cRegionRankSlot
# :call: v core::core::structs::cRegionRankSlots
cdef void cregion_rank_slots_extend(cRegionRankSlots* slots):
    cdef int nmax_old = slots.max
    cdef int nmax_new = 2*nmax_old
    cdef cRegionRankSlot* slots_tmp
    if nmax_old == 0:
        nmax_new = 5
    else:
        slots_tmp = <cRegionRankSlot*>malloc(nmax_old*sizeof(cRegionRankSlot))
        for i in range(nmax_old):
            slots_tmp[i].rank = slots.slots[i].rank
            slots_tmp[i].region = slots.slots[i].region
        free(slots.slots)
    slots.slots = <cRegionRankSlot*>malloc(nmax_new*sizeof(cRegionRankSlot))
    for i in range(nmax_old):
        slots.slots[i].rank = slots_tmp[i].rank
        slots.slots[i].region = slots_tmp[i].region
    for i in range(nmax_old, nmax_new):
        slots.slots[i].rank = 0
        slots.slots[i].region = NULL
    if nmax_old > 0:
        free(slots_tmp)
    slots.max = nmax_new


# :call: > --- CALLERS ---
# :call: > core::tables::pixel_region_table_alloc_pixels
# :call: > core::tables::pixel_region_table_init_regions
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cRegionRankSlot
cdef inline void pixel_region_table_alloc_pixel(
    PixelRegionTable table,
    np.int32_t x,
    np.int32_t y,
    int n_slots,
) nogil:
    if table[x][y].max < n_slots:
        free(table[x][y].slots)
        table[x][y].max = 0
    if table[x][y].max == 0:
        table[x][y].slots = <cRegionRankSlot*>malloc(n_slots*sizeof(cRegionRankSlot))
        table[x][y].max = n_slots
    table[x][y].n = 0
    for i_slot in range(table[x][y].max):
        table[x][y].slots[i_slot].region = NULL
        table[x][y].slots[i_slot].rank = -1


# :call: > --- CALLERS ---
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::csplit_regiongrow_core
# :call: > core::tracking::FeatureTracker::extend_tracks
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cPixel
# :call: v core::core::structs::cRegion
# :call: v core::core::structs::cRegions
# :call: v core::core::tables::pixel_region_table_alloc_pixel
cdef void pixel_region_table_init_regions(
    PixelRegionTable table,
    cRegions* cregions_pixels,
    cRegions* cregions_target,
    int n_slots_max,
):
    # print(f"< pixel_region_table_init_regions {cregions.n}")
    cdef:
        int i_region, i_pixel
        cRegion* cregion_pixels
        cRegion* cregion_target
        cPixel* cpixel
        np.int32_t x, y
    # for i_region in prange(cregions_pixels.n, nogil=True):
    for i_region in range(cregions_pixels.n):
        cregion_pixels = cregions_pixels.regions[i_region]
        cregion_target = cregions_target.regions[
            cregions_target.n - cregions_pixels.n + i_region
        ]
        for i_pixel in range(cregion_pixels.pixels_istart,
                cregion_pixels.pixels_iend):
            if cregion_pixels.pixels[i_pixel] is NULL:
                continue
            x = cregion_pixels.pixels[i_pixel].x
            y = cregion_pixels.pixels[i_pixel].y

            # Allocate or increase slots if necessary
            pixel_region_table_alloc_pixel(table, x, y, n_slots_max)

            # Initialize slots
            # print(f"table[{x}][{y}][0] = {i_region}")
            table[x][y].slots[0].region = cregion_target
            table[x][y].slots[0].rank = -1
            table[x][y].n = 1


# :call: > --- CALLERS ---
# :call: > core::identification::csplit_regiongrow_levels_core
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cRegion
# :call: v core::core::tables::pixel_region_table_alloc_pixels
cdef void pixel_region_table_grow(
    PixelRegionTable table, cRegion* cregion, int n_slots_new):
    # print("< pixel_region_table_grow")
    pixel_region_table_alloc_pixels(table, n_slots_new, cregion)
    # print("> pixel_region_table_grow")


# :call: > --- CALLERS ---
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cRegion
# :call: v core::core::tables::_pixel_region_table_cleanup_entry
cdef void pixel_region_table_cleanup_pixels(PixelRegionTable table, cRegion* cregion):
    # print("< pixel_region_table_cleanup_pixels")
    cdef:
        int i_pixel
        np.int32_t x, y
    for i_pixel in prange(cregion.pixels_max, nogil=True):
        if cregion.pixels[i_pixel] is not NULL:
            _pixel_region_table_cleanup_entry(
                table, cregion.pixels[i_pixel].x, cregion.pixels[i_pixel].y,
            )
    # print("> pixel_region_table_cleanup_pixels")


# :call: > --- CALLERS ---
# :call: > core::identification::cregions2features_connected2neighbors
# :call: > core::identification::find_features_2d_threshold
# :call: > core::typedefs::Grid::reset_tables
# :call: > core::typedefs::grid_reset
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::tables::pixel_region_table_reset_slots
cdef void pixel_region_table_reset(
    PixelRegionTable table, np.int32_t nx, np.int32_t ny,
):
    cdef np.int32_t x
    cdef np.int32_t y
    for x in prange(nx, nogil=True):
        for y in range(ny):
            pixel_region_table_reset_slots(table, x, y)


# :call: > --- CALLERS ---
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::csplit_regiongrow_core
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::tables::pixel_region_table_reset_regions
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cRegion
# :call: v core::core::tables::pixel_region_table_reset_slots
cdef void pixel_region_table_reset_region(PixelRegionTable table, cRegion* cregion):
    cdef int i
    for i in prange(cregion.pixels_istart, cregion.pixels_iend, nogil=True):
        if cregion.pixels[i] is not NULL:
            pixel_region_table_reset_slots(
                table, cregion.pixels[i].x, cregion.pixels[i].y,
            )


# :call: > --- CALLERS ---
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::structs::cRegions
# :call: v core::core::tables::pixel_region_table_reset_region
cdef void pixel_region_table_reset_regions(PixelRegionTable table, cRegions* cregions):
    cdef int i
    for i in range(cregions.n):
        pixel_region_table_reset_region(table, cregions.regions[i])


# :call: > --- CALLERS ---
# :call: > core::identification::regiongrow_resolve_multi_assignments
# :call: > core::tables::pixel_region_table_reset
# :call: > core::tables::pixel_region_table_reset_region
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::tables::cregion_rank_slots_reset
@cython.profile(False)
cdef inline void pixel_region_table_reset_slots(
    PixelRegionTable table, np.int32_t x, np.int32_t y,
) nogil:
    cregion_rank_slots_reset(&table[x][y])


# :call: > --- CALLERS ---
# :call: > core::identification::resolve_multi_assignment_best_connected_region
# :call: > core::identification::resolve_multi_assignment_biggest_region
# :call: > core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > core::identification::resolve_multi_assignment_strongest_region
# :call: > core::tables::pixel_region_table_reset_slots
# :call: v --- CALLING ---
@cython.profile(False)
cdef void cregion_rank_slots_reset(cRegionRankSlots* slots) nogil:
    cdef int i_slot
    for i_slot in range(slots.max):
        slots.slots[i_slot].region = NULL
        slots.slots[i_slot].rank = -1
    slots.n = 0


# :call: > --- CALLERS ---
# :call: > core::identification::resolve_multi_assignment_biggest_region
# :call: > core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > core::identification::resolve_multi_assignment_strongest_region
# :call: v --- CALLING ---
# :call: v core::core::structs::cRegionRankSlot
# :call: v core::core::structs::cRegionRankSlots
cdef cRegionRankSlots cregion_rank_slots_copy(cRegionRankSlots* slots):
    cdef cRegionRankSlots slots2
    slots2.slots = <cRegionRankSlot*>malloc(slots.max*sizeof(cRegionRankSlot))
    slots2.max = slots.max
    slots2.n = slots.n
    for i in range(slots.max):
        slots2.slots[i].region = slots.slots[i].region
        slots2.slots[i].rank = slots.slots[i].rank
    return slots2


# :call: > --- CALLERS ---
# :call: > core::tables::pixel_region_table_cleanup
# :call: > core::tables::pixel_region_table_cleanup_pixels
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
cdef void _pixel_region_table_cleanup_entry(PixelRegionTable table, int x, int y) nogil:
    if table[x][y].max > 0:
        free(table[x][y].slots)
        table[x][y].slots = NULL
        table[x][y].max = 0
        table[x][y].n = 0


# :call: > --- CALLERS ---
# :call: > core::typedefs::grid_cleanup
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelRegionTable
# :call: v core::core::tables::_pixel_region_table_cleanup_entry
cdef void pixel_region_table_cleanup(
    PixelRegionTable table, np.int32_t nx, np.int32_t ny,
):
    # print("< pixel_region_table_cleanup")
    cdef int i
    cdef int j
    if table is not NULL:
        for i in prange(nx, nogil=True):
            for j in range(ny):
                _pixel_region_table_cleanup_entry(table, i, j)
            free(table[i])
        free(table)
    # print("> pixel_region_table_cleanup")


# :call: > --- CALLERS ---
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::csplit_regiongrow_core
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelStatusTable
# :call: v core::core::structs::cPixel
# :call: v core::core::structs::cRegion
# :call: v core::core::structs::cRegions
cdef void pixel_status_table_init_feature(
    PixelStatusTable table, cRegion* cfeature, cRegions* cregions_seeds,
):
    """Initialize pixel status table from feature and seeds.

    Codes:
     * -2: background
     * -1: seed feature
     *  0: definitively assigned in earlier iteration
     *  1: definitively assigned in this iteration
     *  2: provisionally assigned in this iteration
     *  3: unassigned

    TODO consider marking boundary pixels separately (of feature/seeds)

    """
    # print("< pixel_status_table_init_feature")
    cdef int i
    cdef int j
    cdef cPixel* cpixel
    cdef cRegion* cregion

    # Initialize feature pixels (to be assigned)
    for i in prange(cfeature.pixels_max, nogil=True):
        cpixel = cfeature.pixels[i]
        if cpixel is NULL:
            continue
        table[cpixel.x][cpixel.y] = 3

    # Initialize seed pixels
    for i in prange(cregions_seeds.n, nogil=True):
        cregion = cregions_seeds.regions[i]
        for j in range(cregion.pixels_max):
            cpixel = cregion.pixels[j]
            if cpixel is not NULL:
                table[cpixel.x][cpixel.y] = -1

    # print("> pixel_status_table_init_feature")


# :call: > --- CALLERS ---
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::csplit_regiongrow_core
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelStatusTable
# :call: v core::core::structs::cRegion
cdef void pixel_status_table_reset_feature(PixelStatusTable table, cRegion* cfeature):
    # print("< pixel_status_table_reset_feature")
    cdef int i_pixel
    cdef np.int32_t x
    cdef np.int32_t y
    for i_pixel in prange(cfeature.pixels_max, nogil=True):
        if cfeature.pixels[i_pixel] is NULL:
            continue
        x = cfeature.pixels[i_pixel].x
        y = cfeature.pixels[i_pixel].y
        table[x][y] = -2
    # print("> pixel_status_table_reset_feature")


# :call: > --- CALLERS ---
# :call: > core::identification::_find_features_threshold_random_seeds
# :call: > core::identification::c_find_features_2d_threshold_seeds
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::identification::feature_split_regiongrow
# :call: > core::identification::find_features_2d_threshold
# :call: > core::typedefs::Grid::__cinit__
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelStatusTable
# :call: v core::core::structs::cConstants
cdef void pixel_status_table_alloc(PixelStatusTable* table, cConstants* constants):
    cdef int i
    cdef int j
    if table[0] is not NULL:
        err = "pixel_status_table_alloc: table must be NULL"
        raise Exception(err)
    table[0] = <PixelStatusTable>malloc(constants.nx*sizeof(np.int8_t*))
    for i in prange(constants.nx, nogil=True):
        table[0][i] = <np.int8_t*>malloc(constants.ny*sizeof(np.int8_t))
        for j in range(constants.ny):
            table[0][i][j] = -2


# :call: > --- CALLERS ---
# :call: > core::typedefs::Grid::reset_tables
# :call: > core::typedefs::grid_reset
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelStatusTable
cdef void pixel_status_table_reset(
    PixelStatusTable table, np.int32_t nx, np.int32_t ny,
):
    cdef np.int32_t x
    cdef np.int32_t y
    for x in prange(nx, nogil=True):
        for y in range(ny):
            table[x][y] = -2


# :call: > --- CALLERS ---
# :call: > core::typedefs::grid_cleanup
# :call: v --- CALLING ---
# :call: v core::core::structs::PixelStatusTable
cdef void pixel_status_table_cleanup(PixelStatusTable table, np.int32_t nx):
    cdef int i
    for i in prange(nx, nogil=True):
        free(table[i])
    free(table)


# :call: > --- CALLERS ---
# :call: > core::identification::_find_features_threshold_random_seeds
# :call: > core::identification::c_find_features_2d_threshold_seeds
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::identification::feature_split_regiongrow
# :call: > core::identification::features_find_neighbors
# :call: > core::identification::find_features_2d_threshold
# :call: > core::identification::merge_adjacent_features
# :call: > core::identification::pixels_find_boundaries
# :call: > core::typedefs::Grid::__cinit__
# :call: v --- CALLING ---
# :call: v core::core::structs::NeighborLinkStatTable
# :call: v core::core::structs::cConstants
cdef void neighbor_link_stat_table_alloc(
    NeighborLinkStatTable* table, cConstants* constants,
) except *:
    # print("< neighbor_link_stat_table_alloc")
    cdef int i
    cdef int j
    cdef int k
    if table[0] is not NULL:
        err = "neighbor_link_stat_table_alloc: table must be NULL"
        raise Exception(err)
        # with cython.cdivision(True): i = 5 / 0  # for valgrind
    table[0] = <np.int8_t***>malloc(constants.nx*sizeof(np.int8_t**))
    for i in prange(constants.nx, nogil=True):
        table[0][i] = <np.int8_t**>malloc(
                constants.ny*sizeof(np.int8_t*))
        for j in range(constants.ny):
            table[0][i][j] = <np.int8_t*>malloc(
                    constants.n_neighbors_max*sizeof(np.int8_t))
            for k in range(constants.n_neighbors_max):
                table[0][i][j][k] = -2


# :call: > --- CALLERS ---
# :call: > core::identification::find_features_2d_threshold
# :call: > core::typedefs::Grid::reset_tables
# :call: > core::typedefs::_reconstruct_boundaries
# :call: > core::typedefs::grid_reset
# :call: v --- CALLING ---
# :call: v core::core::structs::NeighborLinkStatTable
# :call: v core::core::structs::cConstants
cdef void neighbor_link_stat_table_reset(
    NeighborLinkStatTable table, cConstants* constants,
) except *:
    # print("< neighbor_link_stat_table_reset")
    cdef int i
    cdef int j
    cdef int k
    for i in prange(constants.nx, nogil=True):
        for j in range(constants.ny):
            for k in range(constants.n_neighbors_max):
                table[i][j][k] = -2


# :call: > --- CALLERS ---
# :call: > core::identification::csplit_regiongrow_levels_core
# :call: v --- CALLING ---
# :call: v core::core::structs::NeighborLinkStatTable
# :call: v core::core::structs::cRegion
cdef void neighbor_link_stat_table_reset_pixels(
    NeighborLinkStatTable table, cRegion* cregion, int n_neighbors_max,
) except *:
    # print("< neighbor_link_stat_table_reset_pixels")
    cdef:
        int i, j, k
        np.int32_t x, y
    for i in prange(cregion.pixels_istart, cregion.pixels_iend, nogil=True):
        if cregion.pixels[i] is not NULL:
            x = cregion.pixels[i].x
            y = cregion.pixels[i].y
            for j in range(n_neighbors_max):
                table[x][y][j] = -2


# :call: > --- CALLERS ---
# :call: > core::typedefs::grid_cleanup
# :call: v --- CALLING ---
# :call: v core::core::structs::NeighborLinkStatTable
cdef void neighbor_link_stat_table_cleanup(
    NeighborLinkStatTable table, np.int32_t nx, np.int32_t ny,
) except *:
    # print("< neighbor_link_stat_table_cleanup")
    cdef int i
    cdef int j
    for i in prange(nx, nogil=True):
        for j in range(ny):
            free(table[i][j])
        free(table[i])
    free(table)


# :call: > --- CALLERS ---
# :call: > core::typedefs::_reconstruct_boundaries
# :call: v --- CALLING ---
# :call: v core::core::structs::NeighborLinkStatTable
# :call: v core::core::structs::cConstants
# :call: v core::core::structs::cPixel
# :call: v core::core::structs::cRegion
# :call: v core::core::structs::get_matching_neighbor_id
# :call: v core::core::tables::get_direct_neighbor_index
cdef void neighbor_link_stat_table_init(
    NeighborLinkStatTable table, cRegion* boundary_pixels, cConstants* constants,
) except *:
    """Initialize table to keep track of checked neighbors.

    Codes:
    - -2: background pixel
    - -1: interior feature pixel
    -  0: link already passed in this direction
    -  1: link already passed in opposite direction
    -  2: unpassed link to other boundary pixel
    """
    cdef bint debug = False
    if debug:
        log.debug(
            f"< neighbor_link_stat_table_init (boundary_pixels: "
            f"id={boundary_pixels.id}, pixels_n={boundary_pixels.pixels_n})"
        )
    cdef int i
    cdef int j
    cdef int k
    cdef int l
    cdef cPixel* cpixel
    cdef cPixel* cpixel2
    cdef np.int32_t x
    cdef np.int32_t y
    cdef np.int32_t x2
    cdef np.int32_t y2
    cdef int ind
    cdef int ind_2
    cdef int ind_left
    cdef int ind_right
    for i in range(boundary_pixels.pixels_istart, boundary_pixels.pixels_iend):
        cpixel = boundary_pixels.pixels[i]
        if cpixel is not NULL:
            x = cpixel.x
            y = cpixel.y
        if debug:
            log.debug(f"{i} ({x}, {y})")

        # Mark all links to other boundary pixels with '2'
        for ind in range(constants.n_neighbors_max):
            other_cpixel = cpixel.neighbors[ind]
            if other_cpixel is NULL:
                if debug:
                    log.debug(f" ind {ind}: other_cpixel is NULL")
                # outside the domain
                continue

            # Other pixel is not a boundary feature, i.e. either unassigned,
            # assigned to a different feature, or interior of same feature
            if not other_cpixel.is_feature_boundary:
                if other_cpixel.region is NULL:
                    # not assigned to a feature
                    if debug:
                        log.debug(
                            f"table[{x}][{y}][{ind}] == {table[x][y][ind]} (no feature)"
                        )
                    continue
                if other_cpixel.region.id != cpixel.region.id:
                    # assigned to a different feature
                    if debug:
                        log.debug(
                            f"table[{x}][{y}][{ind}] == {table[x][y][ind]} "
                            f"({other_cpixel.x}, {other_cpixel.y})"
                            f"[{other_cpixel.region.id}] (different feature)"
                        )
                    continue
                # non-boundary (i.e. interior) pixel of the same feature
                if debug:
                    log.debug(
                        f"table[{x}][{y}][{ind}] =  -1 ({other_cpixel.x}, "
                        f"{other_cpixel.y})[{other_cpixel.region.id}] (interior)"
                    )
                table[x][y][ind] = -1
                continue

            # Other pixel is a boundary feature of the same feature
            if other_cpixel.region.id == cpixel.region.id:
                if debug:
                    log.debug(
                        f"table[{x}][{y}][{ind}] =   2 ({other_cpixel.x}, "
                        f"{other_cpixel.y})[{other_cpixel.region.id}] (same boundary)"
                    )
                table[x][y][ind] = 2
                continue

            if debug:
                rid = "-" if other_cpixel.region is NULL else other_cpixel.region.id
                log.debug(
                    f"table[{x}][{y}][{ind}] == {table[x][y][ind]} "
                    f"({other_cpixel.x}, {other_cpixel.y})[{rid}] (else-case)"
                )

    # Remove unwanted links
    for i in range(boundary_pixels.pixels_istart, boundary_pixels.pixels_iend):
        cpixel = boundary_pixels.pixels[i]
        if cpixel is NULL:
            continue
        x = cpixel.x
        y = cpixel.y

        for ind in range(constants.n_neighbors_max):
            if constants.connectivity == 4 and ind%2 > 0:
                continue
            if table[x][y][ind] < 2:
                continue
            cpixel2 = cpixel.neighbors[ind]
            ind_2 = get_matching_neighbor_id(ind, constants.n_neighbors_max)
            x2 = cpixel2.x
            y2 = cpixel2.y

            # Identify links through the inside of the feature
            # All four neighboring links (both directions) are links
            # First check the forward direction (from this pixel)
            ind_left = get_direct_neighbor_index(-1, ind, constants.connectivity)
            ind_right = get_direct_neighbor_index(1, ind, constants.connectivity)
            if table[x][y][ind_left] >= -1 and table[x][y][ind_right] >= -1:
                # Now also check the backward direction (from other pixel)
                ind_left = get_direct_neighbor_index(-1, ind_2, constants.connectivity)
                ind_right = get_direct_neighbor_index(1, ind_2, constants.connectivity)
                if table[x2][y2][ind_left] >= -1 and table[x2][y2][ind_right] >= -1:
                    if debug:
                        log.debug(f"table[{x}][{y}][{ind}] = 0")
                    table[x][y][ind] = 0
                    continue

            # SR_TODO test if this check makes the one above obsolete
            # A link is only valid if there is at least one non-feature pixel
            # to the left of it (for clockwise search direction)
            ind_left = get_direct_neighbor_index(-1, ind, constants.connectivity)
            if table[x][y][ind_left] > -2:
                if cpixel2 is not NULL:
                    ind_2 = get_matching_neighbor_id(ind, constants.n_neighbors_max)
                    ind_right = get_direct_neighbor_index(
                        1, ind_2, constants.connectivity,
                    )
                    if table[x2][y2][ind_right] > -2:
                        if debug:
                            log.debug(f"table[{x}][{y}][{ind}] = 0")
                        table[x][y][ind] = 0
                        continue

            # 8-connectivity links where a diagnoal connection is preferred
            if constants.connectivity == 8 and ind%2 == 0:
                ind_left = get_direct_neighbor_index(-1, ind, 4)
                ind_right = get_direct_neighbor_index(1, ind, 4)

                # A diagonal link is prefered if there are feature pixels
                # to the start-left and end-right or start-right and end-left
                if (
                    (table[x][y][ind_left] >= 0 and table[x2][y2][ind_right] >= 0)
                    or (table[x][y][ind_right] >= 0 and table[x2][y2][ind_left] >= 0)
                ):
                    if debug:
                        log.debug(f"table[{x}][{y}][{ind}] = 0")
                    table[x][y][ind] = 0
                    continue

                # A diagnoal link is preferred if there are feature pixels
                # on both sides of the start or end of the link
                if (
                    (table[x][y][ind_left] > 0 and table[x][y][ind_right] > 0)
                    or (table[x2][y2][ind_left] > 0 and table[x2][y2][ind_right] > 0)
                ):
                    if debug:
                        log.debug(f"table[{x}][{y}][{ind}] = 0")
                    table[x][y][ind] = 0
                    continue


# :call: > --- CALLERS ---
# :call: > core::tables::neighbor_link_stat_table_init
# :call: v --- CALLING ---
cdef inline int get_direct_neighbor_index(
    int direction, int ind, int connectivity
) nogil:
    if connectivity == 4:
        if direction < 0:
            if ind == 0:
                return 6
            return ind - 2
        elif direction > 0:
            if ind == 6:
                return 0
            return ind + 2
    if connectivity == 8:
        if direction < 0:
            if ind == 0:
                return 7
            return ind - 1
        elif direction > 0:
            if ind == 7:
                return 0
            return ind + 1
    with gil:
        log.error("direct neighbor index not found")
    exit(44)
