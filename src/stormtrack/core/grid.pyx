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

# Standard library
import logging as log

# Third-party
import numpy as np


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::identify_features
# :call: > stormtrack::core::tracking::FeatureTracker::__cinit__
# :call: > stormtrack::core::tracking::FeatureTracker::_swap_grids
# :call: > stormtrack::track_features::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: v stormtrack::core::tables::pixel_region_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_reset
# :call: v stormtrack::core::tables::pixel_status_table_alloc
# :call: v stormtrack::core::tables::pixel_status_table_reset
# :call: v stormtrack::core::constants::Constants
# :call: v stormtrack::core::grid::grid_cleanup
# :call: v stormtrack::core::grid::grid_create
# :call: v stormtrack::core::grid::grid_reset
# :call: v stormtrack::core::grid::grid_set_values
cdef class Grid:
    def __cinit__(self,
        Constants constants,
        np.float32_t val = 0,
        bint alloc_tables = False,
        int n_slots = -1,
    ):
        fld = np.full([constants.nx, constants.ny], val, np.float32)
        self._cgrid = grid_create(fld, constants.to_c()[0])
        self.constants = constants

        # SR_TMP < TODO cleaner implementation
        if alloc_tables:
            if n_slots < 0:
                raise ValueError("must pass n_slots >= 0 to alloc tables")
            neighbor_link_stat_table_alloc(
                &self._cgrid.neighbor_link_stat_table, &self._cgrid.constants,
            )
            pixel_status_table_alloc(
                &self._cgrid.pixel_status_table, &self._cgrid.constants,
            )
            pixel_region_table_alloc(
                &self._cgrid.pixel_region_table, n_slots, &self._cgrid.constants,
            )
        # SR_TMP >

    def __dealloc__(self):
        grid_cleanup(&self._cgrid)

    cpdef void set_values(self, np.ndarray[np.float32_t, ndim=2] fld):
        grid_set_values(&self._cgrid, fld)

    cpdef void reset(self):
        grid_reset(&self._cgrid)

    cdef cGrid* to_c(self):
        return &(self._cgrid)

    cdef void reset_tables(self):
        if self._cgrid.neighbor_link_stat_table is not NULL:
            neighbor_link_stat_table_reset(
                self._cgrid.neighbor_link_stat_table, &self._cgrid.constants,
            )
        if self._cgrid.pixel_status_table is not NULL:
            pixel_status_table_reset(
                self._cgrid.pixel_status_table, self.constants.nx, self.constants.ny,
            )
        if self._cgrid.pixel_region_table is not NULL:
            pixel_region_table_reset(
                self._cgrid.pixel_region_table, self.constants.nx, self.constants.ny,
            )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::grid::Grid::__cinit__
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::grid_create_empty
# :call: v stormtrack::core::grid::grid_create_pixels
cdef cGrid grid_create(np.float32_t[:, :] fld, cConstants constants) except *:
    # print("< grid_create")  # SR_DBG
    cdef cGrid grid = grid_create_empty(constants)
    grid_create_pixels(&grid, fld)
    return grid


# :call: > --- CALLING ---
# :call: > stormtrack::core::structs::cGrid
# :call: > stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: > stormtrack::core::tables::pixel_region_table_reset
# :call: > stormtrack::core::tables::pixel_status_table_reset
# :call: > stormtrack::core::cpixel::cpixels_reset
# :call: > stormtrack::core::cregions_store::cregions_store_reset
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- CALLERS ---
# :call: v stormtrack::core::grid::Grid::reset
cdef void grid_reset(cGrid* grid) except *:
    cpixels_reset(grid.pixels, grid.constants.nx, grid.constants.ny)
    cregions_store_reset(&grid._regions)
    if grid.pixel_region_table is not NULL:
        pixel_region_table_reset(
            grid.pixel_region_table, grid.constants.nx, grid.constants.ny,
        )
    if grid.pixel_status_table is not NULL:
        pixel_status_table_reset(
            grid.pixel_status_table, grid.constants.nx, grid.constants.ny,
        )
    if grid.pixel_done_table is not NULL:
        log.error("not implemented: grid_reset/pixel_done_table_reset")
        exit(4)
        # pixel_done_table_reset(grid.pixel_done_table, cregion)
    if grid.neighbor_link_stat_table is not NULL:
        neighbor_link_stat_table_reset(
            grid.neighbor_link_stat_table, &grid.constants,
        )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::grid::Grid::__dealloc__
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::tables::neighbor_link_stat_table_cleanup
# :call: v stormtrack::core::tables::pixel_done_table_cleanup
# :call: v stormtrack::core::tables::pixel_region_table_cleanup
# :call: v stormtrack::core::tables::pixel_status_table_cleanup
# :call: v stormtrack::core::cregions_store::cregions_store_cleanup
cdef void grid_cleanup(cGrid* grid) except *:
    # print("< GRID CLEANUP") # SR_DBG
    grid.timestep = 0

    cdef int i
    if grid.pixels is not NULL:
        for i in range(grid.constants.nx):
            free(grid.pixels[i])
        free(grid.pixels)
        grid.pixels = NULL

    if grid.pixel_region_table is not NULL:
        pixel_region_table_cleanup(
            grid.pixel_region_table, grid.constants.nx, grid.constants.ny,
        )
        grid.pixel_region_table = NULL

    if grid.pixel_status_table is not NULL:
        pixel_status_table_cleanup(grid.pixel_status_table, grid.constants.nx)
        grid.pixel_status_table = NULL

    if grid.pixel_done_table is not NULL:
        pixel_done_table_cleanup(grid.pixel_done_table, &grid.constants)
        grid.pixel_done_table = NULL

    if grid.neighbor_link_stat_table is not NULL:
        neighbor_link_stat_table_cleanup(
            grid.neighbor_link_stat_table, grid.constants.nx, grid.constants.ny,
        )
        grid.neighbor_link_stat_table = NULL

    cregions_store_cleanup(&grid._regions)


# :call: > --- callers ---
# :call: > stormtrack::core::grid::grid_create
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::cregion::_collect_neighbors
# :call: v stormtrack::core::cpixel::cpixel2d_create
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void grid_create_pixels(cGrid* grid, np.float32_t[:, :] fld) except *:
    cdef bint debug=False
    if debug:
        log.debug(f"< grid_create_pixels {grid.constants.nx}x{grid.constants.ny}")
    cdef int i
    cdef int j
    cdef int k
    cdef int n_neigh
    cdef cPixel* cpixel

    # Create pixels
    grid.pixels = <cPixel**>malloc(grid.constants.nx*sizeof(cPixel*))
    for i in prange(grid.constants.nx, nogil=True):
        grid.pixels[i] = cpixel2d_create(grid.constants.ny)
        for j in range(grid.constants.ny):
            cpixel = &grid.pixels[i][j]
            cpixel.id = i*grid.constants.ny + j
            cpixel.x = i
            cpixel.y = j
            cpixel.v = fld[i, j]
            cpixel.connectivity = grid.constants.connectivity

    # Determine numbers of neighbors
    for i in range(grid.constants.nx):
        for j in range(grid.constants.ny):
            cpixel = &grid.pixels[i][j]
            n_neigh = _collect_neighbors(
                i,
                j,
                cpixel.neighbors,
                grid.pixels,
                &grid.constants,
                grid.constants.connectivity,
            )
            cpixel.neighbors_n = n_neigh


# :call: > --- callers ---
# :call: > stormtrack::core::grid::Grid::set_values
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void grid_set_values(cGrid* grid, np.float32_t[:, :] fld) except *:
    cdef bint debug=False
    if debug:
        log.debug(f"< grid_set_values {grid.constants.nx}x{grid.constants.ny}")
    cdef int i
    cdef int j
    cdef int k
    cdef int n_neigh
    cdef cPixel* cpixel
    for i in prange(grid.constants.nx, nogil=True):
        for j in range(grid.constants.ny):
            cpixel = &grid.pixels[i][j]
            cpixel.v = fld[i, j]


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
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregions_store::cregions_store_get_new_region
cdef cRegion* grid_create_cregion(cGrid* grid) except *:
    return cregions_store_get_new_region(&grid._regions)
