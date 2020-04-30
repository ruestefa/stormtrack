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
from cython cimport boundscheck
from cython cimport cdivision
from cython cimport profile
from cython cimport wraparound
from cython.parallel cimport prange

# Standard library
import logging as log

# Third-party
import numpy as np

# sign
# default_constants
# cregion_get_unique_id
# Constants
# cregion_init
# cregion_insert_pixels_coords
# _cregion_create_pixels
# cregion_insert_pixel
# cregion_insert_pixel_nogil
# cregion_remove_pixel
# cregion_remove_pixel_nogil
# _cregion_remove_pixel_from_pixels
# _cregion_remove_pixel_from_pixels_nogil
# _cregion_remove_pixel_from_shells
# _cregion_remove_pixel_from_shells_nogil
# _cregion_remove_pixel_from_holes
# _cregion_remove_pixel_from_holes_nogil
# cregion_pixels_remove_gaps
# cregion_pixels_remove_gaps_nogil
# _cregion_shell_remove_gaps
# _cregion_shell_remove_gaps_nogil
# _cregion_hole_remove_gaps
# _cregion_hole_remove_gaps_nogil
# _cregion_insert_shell_pixel
# _cregion_insert_hole_pixel
# _cregion_reconnect_pixel
# _cregion_extend_pixels
# _cregion_extend_pixels_nogil
# _cregion_extend_shell
# _cregion_extend_hole
# _cregion_extend_shells
# _cregion_extend_holes
# _cregion_new_shell
# _cregion_new_hole
# _cregion_add_connected
# _cregion_extend_connected
# cregion_reset
# _cpixel_unlink_region
# _cregion_reset_connected
# cregion_remove_connected
# cregion_cleanup
# cregion_merge
# cregion_determine_boundaries
# cregions_determine_boundaries
# _cregion_determine_boundaries_core
# _reconstruct_boundaries
# _find_link_to_continue
# categorize_boundaries
# neighbor_pixel_angle
# _extract_closed_path
# categorize_boundary_is_shell
# boundary_must_be_a_shell
# cregion_check_validity
# cregion_northernmost_pixel
# cregions_find_northernmost_uncategorized_region
# cregion_reset_boundaries
# _determine_boundary_pixels_raw
# _collect_neighbors
# cpixel_get_neighbor
# _cpixel_get_neighbor
# cregion_overlaps
# cregion_overlaps_tables
# cregion_overlap_n
# cregion_overlap_n_tables
# _cregion_overlap_core
# cregion_overlap_n_mask
# cregion_determine_bbox
# cregions_get_unique_id
# cregions_init
# cregions_create
# cregions_link_region
# cregions_extend
# cregions_move
# cregions_reset
# cregions_cleanup
# cregions_connect
# cregions_find_connected
# dbg_check_connected
# cpixel_set_region
# cpixels_reset
# cpixel2d_create
# cregions_store_get_new_region
# cregions_store_reset
# cregions_store_extend
# cregions_store_cleanup
# Grid
# grid_create
# grid_reset
# grid_cleanup
# grid_create_pixels
# grid_set_values
# grid_new_region
# grid_new_regions


# CALL_TODO stormtrack::extra::front_surgery


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::categorize_boundaries
# :call: v --- CALLING ---
cdef inline int sign(int num):
    if num >= 0:
        return 1
    else:
        return -1


# :call: > --- CALLERS ---
# :call: > stormtrack::identify_features::*
# :call: > stormtrack::core::identification::Feature::derive_boundaries_from_pixels
# :call: > stormtrack::core::identification::Feature::derive_holes_from_pixels
# :call: > stormtrack::core::identification::Feature::derive_shells_from_pixels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::find_features_2d_threshold_seeded
# :call: > stormtrack::core::identification::identify_features
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::split_regiongrow_levels
# :call: > stormtrack::track_features::*
# :call: > test_stormtrack::test_core::test_features::test_boundaries::*
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: > test_stormtrack::test_core::test_features::test_regions::*
# :call: > test_stormtrack::test_core::test_features::test_split_regiongrow::*
# :call: > test_stormtrack::utils::*
# :call: v --- CALLING ---
# :call: v stormtrack::core::typedefs::Constants
cpdef Constants default_constants(
    # SR_TODO remove nx, ny (use from Grid)
    int nx, int ny, int connectivity=4, int n_neighbors_max=8,
):
    return Constants(
        nx=nx, ny=ny, connectivity=connectivity, n_neighbors_max=n_neighbors_max,
    )


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::typedefs::_determine_boundary_pixels_raw
# :call: > stormtrack::core::typedefs::cregions_store_extend
# :call: v --- CALLING ---
cdef np.uint64_t cregion_get_unique_id():
    global CREGION_NEXT_ID
    cdef np.uint64_t rid = CREGION_NEXT_ID
    CREGION_NEXT_ID += 1
    return rid


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::find_features_2d_threshold_seeded
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::tracking::FeatureTrack::merge_features
# :call: > stormtrack::core::tracking::FeatureTracker::__cinit__
# :call: > stormtrack::core::typedefs::Grid::__cinit__
# :call: > stormtrack::core::typedefs::default_constants
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cConstants
cdef class Constants:

    def __cinit__(self,
        # SR_TMP < TODO remove (use from Grid)
        int nx,
        int ny,
        # SR_TMP >
        int connectivity,
        int n_neighbors_max,
    ):
        self.nx = nx
        self.ny = ny
        self.connectivity = connectivity
        self.n_neighbors_max = n_neighbors_max
        self._cconstants = cConstants(
            nx=nx, ny=ny, connectivity=connectivity, n_neighbors_max=n_neighbors_max,
        )

    cdef cConstants* to_c(self):
        return &(self._cconstants)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::typedefs::_determine_boundary_pixels_raw
# :call: > stormtrack::core::typedefs::cregions_store_extend
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
cdef void cregion_init(cRegion* cregion, cRegionConf cregion_conf, np.uint64_t rid):
    cdef int i
    cdef int j
    # print("< cregion_init")

    cregion.id = rid

    # Connected regions
    cregion.connected_max = cregion_conf.connected_max
    cregion.connected_n = 0
    cregion.connected = NULL
    if cregion_conf.connected_max > 0:
        cregion.connected = <cRegion**>malloc(
            cregion_conf.connected_max*sizeof(cRegion*),
        )
        for i in prange(cregion_conf.connected_max, nogil=True):
            cregion.connected[i] = NULL

    # Pixels
    cregion.pixels_max = cregion_conf.pixels_max
    cregion.pixels_n = 0
    cregion.pixels_istart = 0
    cregion.pixels_iend = 0
    cregion.pixels = NULL
    if cregion_conf.pixels_max > 0:
        cregion.pixels = <cPixel**>malloc(
            cregion_conf.pixels_max*sizeof(cPixel*),
        )
        for i in prange(cregion_conf.pixels_max, nogil=True):
            cregion.pixels[i] = NULL

    # Shell pixels
    # SR_ONE_SHELL < TODO remove once multiple shells properly implemented
    # -cregion.shell_max = cregion_conf.shell_max
    # -cregion.shell_n = 0
    # -cregion.shell = NULL
    # -if cregion_conf.shell_max > 0:
    # -    cregion.shell = <cPixel**>malloc(cregion_conf.shell_max*sizeof(cPixel*))
    # -    for i in prange(cregion_conf.shell_max, nogil=True):
    # -        cregion.shell[i] = NULL
    # SR_ONE_SHELL >
    cregion.shells_max = cregion_conf.shells_max
    cregion.shells_n = 0
    cregion.shells = NULL
    cregion.shell_n = NULL
    cregion.shell_max = NULL
    if cregion_conf.shells_max > 0:
        cregion.shells = <cPixel***>malloc(cregion_conf.shells_max*sizeof(cPixel**))
        cregion.shell_n = <int*>malloc(cregion_conf.shells_max*sizeof(int))
        cregion.shell_max = <int*>malloc(cregion_conf.shells_max*sizeof(int))
        for i in prange(cregion_conf.shells_max, nogil=True):
            cregion.shell_n[i] = 0
            cregion.shell_max[i] = cregion_conf.shell_max
            cregion.shells[i] = NULL
            if cregion_conf.shell_max > 0:
                cregion.shells[i] = <cPixel**>malloc(
                    cregion_conf.shell_max*sizeof(cPixel*),
                )
                for j in range(cregion_conf.shell_max):
                    cregion.shells[i][j] = NULL

    # Holes pixels
    cregion.holes_max = cregion_conf.holes_max
    cregion.holes_n = 0
    cregion.holes = NULL
    cregion.hole_n = NULL
    cregion.hole_max = NULL
    if cregion_conf.holes_max > 0:
        cregion.holes = <cPixel***>malloc(cregion_conf.holes_max*sizeof(cPixel**))
        cregion.hole_n = <int*>malloc(cregion_conf.holes_max*sizeof(int))
        cregion.hole_max = <int*>malloc(cregion_conf.holes_max*sizeof(int))
        for i in prange(cregion_conf.holes_max, nogil=True):
            cregion.hole_n[i] = 0
            cregion.hole_max[i] = cregion_conf.hole_max
            cregion.holes[i] = NULL
            if cregion_conf.hole_max > 0:
                cregion.holes[i] = <cPixel**>malloc(
                    cregion_conf.hole_max*sizeof(cPixel*),
                )
                for j in range(cregion_conf.hole_max):
                    cregion.holes[i][j] = NULL


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cregion_insert_pixel
@boundscheck(False)
@wraparound(False)
cdef void cregion_insert_pixels_coords(
    cRegion* cregion,
    cPixel** cpixels,
    np.ndarray[np.int32_t, ndim=2] coords,
    bint link_region,
    bint unlink_pixels,
) except *:
    cdef int i
    cdef int n_pixels = coords.shape[0]
    cdef np.int32_t x
    cdef np.int32_t y
    cdef cPixel* cpixel
    # for i in prange(n_pixels, nogil=True):
    for i in range(n_pixels):
        x = coords[i, 0]
        y = coords[i, 1]
        cpixel = &cpixels[x][y]
        cregion_insert_pixel(cregion, cpixel, link_region, unlink_pixels)


# :call: > --- CALLERS ---
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cpixel2d_create
cdef void _cregion_create_pixels(
    cRegion* cregion,
    int istart,
    int iend,
    int connectivity,
):
    cdef int i
    cdef int n=(iend - istart)
    cdef cPixel* cpixel
    cdef cPixel* cpixels = cpixel2d_create(n)
    for i in range(istart, iend):
        cpixel = &cpixels[i+istart]
        cpixel.connectivity = connectivity
        cregion.pixels[i] = cpixel
    cpixels = NULL


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::identification::regiongrow_assign_pixel
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::cregion_insert_pixels_coords
# :call: > stormtrack::core::typedefs::cregion_merge
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_extend_pixels
# :call: v stormtrack::core::typedefs::cpixel_set_region
# :call: v stormtrack::core::typedefs::cregion_remove_pixel
@profile(False)
cdef void cregion_insert_pixel(
    cRegion* cregion,
    cPixel* cpixel,
    bint link_region,
    bint unlink_pixel,
):
    cdef bint debug = False
    if debug:
        log.debug(f"< cregion_insert_pixel [{cregion.id}]<-({cpixel.x}, {cpixel.y})")
    cdef int i_pixel

    # Find empty slot for pixel
    if cregion.pixels_iend >= cregion.pixels_max:
        _cregion_extend_pixels(cregion)
    i_pixel = cregion.pixels_iend

    # Insert pixel
    cregion.pixels[i_pixel] = cpixel
    cregion.pixels_n += 1
    cregion.pixels_iend = i_pixel+1

    # Link region to pixel
    if link_region:
        if unlink_pixel and cpixel.region is not NULL:
            cregion_remove_pixel(cpixel.region, cpixel)
        cpixel_set_region(cpixel, cregion)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_find_background_neighbor_pixels
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_extend_pixels_nogil
# :call: v stormtrack::core::typedefs::cpixel_set_region
# :call: v stormtrack::core::typedefs::cregion_remove_pixel_nogil
@profile(False)
cdef void cregion_insert_pixel_nogil(
    cRegion* cregion, cPixel* cpixel, bint link_region, bint unlink_pixel,
) nogil:
    cdef int i_pixel

    # Find empty slot for pixel
    if cregion.pixels_iend >= cregion.pixels_max:
        _cregion_extend_pixels_nogil(cregion)
    i_pixel = cregion.pixels_iend

    # Insert pixel
    cregion.pixels[i_pixel] = cpixel
    cregion.pixels_n += 1
    cregion.pixels_iend = i_pixel+1

    # Link region to pixel
    if link_region:
        if unlink_pixel and cpixel.region is not NULL:
            cregion_remove_pixel_nogil(cpixel.region, cpixel)
        cpixel_set_region(cpixel, cregion)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::typedefs::cregion_insert_pixel
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_remove_pixel_from_holes
# :call: v stormtrack::core::typedefs::_cregion_remove_pixel_from_pixels
# :call: v stormtrack::core::typedefs::_cregion_remove_pixel_from_shells
@profile(False)
cdef void cregion_remove_pixel(cRegion* cregion, cPixel* cpixel):
    cdef bint debug = False
    if debug:
        log.debug(f"cregion_remove_pixel: [{cregion.id}]</-({cpixel.x}, {cpixel.y})")
    _cregion_remove_pixel_from_pixels(cregion, cpixel)
    _cregion_remove_pixel_from_shells(cregion, cpixel)
    _cregion_remove_pixel_from_holes(cregion, cpixel)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_insert_pixel_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_remove_pixel_from_holes_nogil
# :call: v stormtrack::core::typedefs::_cregion_remove_pixel_from_pixels_nogil
# :call: v stormtrack::core::typedefs::_cregion_remove_pixel_from_shells_nogil
@profile(False)
cdef void cregion_remove_pixel_nogil(cRegion* cregion, cPixel* cpixel) nogil:
    _cregion_remove_pixel_from_pixels_nogil(cregion, cpixel)
    _cregion_remove_pixel_from_shells_nogil(cregion, cpixel)
    _cregion_remove_pixel_from_holes_nogil(cregion, cpixel)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_remove_pixel
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
@profile(False)
cdef inline void _cregion_remove_pixel_from_pixels(cRegion* cregion, cPixel* cpixel):
    # SR_DBG <
    if cregion is NULL:
        log.error("_cregion_remove_pixel_from_pixels: cregion is NULL")
        exit(44)
    if cpixel is NULL:
        log.error("_cregion_remove_pixel_from_pixels: cpixel is NULL")
        exit(44)
    # SR_DBG >
    cdef int i
    cdef cPixel* cpixel_i
    for i in range(cregion.pixels_istart, cregion.pixels_iend):
        cpixel_i = cregion.pixels[i]
        # if cpixel_i is not NULL:
        if cregion.pixels[i] is not NULL:
            if cpixel_i.x == cpixel.x and cpixel_i.y == cpixel.y:
                cregion.pixels[i] = NULL
                cregion.pixels_n -= 1
                return
    for i in range(cregion.pixels_iend):
        if cregion.pixels[i] is not NULL:
            cregion.pixels_istart = i
            break


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_remove_pixel_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
@profile(False)
cdef inline void _cregion_remove_pixel_from_pixels_nogil(
    cRegion* cregion, cPixel* cpixel,
) nogil:
    cdef int i
    cdef cPixel* cpixel_i
    for i in range(cregion.pixels_istart, cregion.pixels_iend):
        cpixel_i = cregion.pixels[i]
        if cpixel_i is not NULL:
            if cpixel_i.x == cpixel.x and cpixel_i.y == cpixel.y:
                cregion.pixels[i] = NULL
                cregion.pixels_n -= 1
                return
    for i in range(cregion.pixels_iend):
        if cregion.pixels[i] is not NULL:
            cregion.pixels_istart = i
            break


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_remove_pixel
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_shell_remove_gaps
@profile(False)
cdef inline void _cregion_remove_pixel_from_shells(cRegion* cregion, cPixel* cpixel):
    cdef int i
    cdef int j
    for j in range(cregion.shells_n):
        for i in range(cregion.shell_n[j]):
            if (
                cregion.shells[j][i].x == cpixel.x
                and cregion.shells[j][i].y == cpixel.y
            ):
                cregion.shells[j][i] = NULL
                _cregion_shell_remove_gaps(cregion, j, i)
                break


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_remove_pixel_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_shell_remove_gaps_nogil
@profile(False)
cdef inline void _cregion_remove_pixel_from_shells_nogil(
    cRegion* cregion, cPixel* cpixel,
) nogil:
    cdef int i
    cdef int j
    for j in range(cregion.shells_n):
        for i in range(cregion.shell_n[j]):
            if (
                cregion.shells[j][i].x == cpixel.x
                and cregion.shells[j][i].y == cpixel.y
            ):
                cregion.shells[j][i] = NULL
                _cregion_shell_remove_gaps_nogil(cregion, j, i)
                break


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_remove_pixel
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_hole_remove_gaps
@profile(False)
cdef inline void _cregion_remove_pixel_from_holes(cRegion* cregion, cPixel* cpixel):
    cdef int i
    cdef int j
    for j in range(cregion.holes_n):
        for i in range(cregion.hole_n[j]):
            if (
                cregion.holes[j][i].x == cpixel.x
                and cregion.holes[j][i].y == cpixel.y
            ):
                cregion.holes[j][i] = NULL
            _cregion_hole_remove_gaps(cregion, j, i)
            return


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_remove_pixel_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_hole_remove_gaps_nogil
@profile(False)
cdef inline void _cregion_remove_pixel_from_holes_nogil(
    cRegion* cregion, cPixel* cpixel,
) nogil:
    cdef int i
    cdef int j
    for j in range(cregion.holes_n):
        for i in range(cregion.hole_n[j]):
            if (
                cregion.holes[j][i].x == cpixel.x
                and cregion.holes[j][i].y == cpixel.y
            ):
                cregion.holes[j][i] = NULL
            _cregion_hole_remove_gaps_nogil(cregion, j, i)
            return


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_extend_pixels
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef inline void cregion_pixels_remove_gaps(cRegion* cregion, int i_start):
    cdef int i
    cdef int j = i_start
    for i in range(i_start, cregion.pixels_max):
        if cregion.pixels[i] is not NULL:
            cregion.pixels[j] = cregion.pixels[i]
            j += 1
    for i in prange(j, cregion.pixels_max, nogil=True):
        cregion.pixels[i] = NULL
    cregion.pixels_istart = 0
    cregion.pixels_iend = cregion.pixels_n


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_extend_pixels_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef inline void cregion_pixels_remove_gaps_nogil(cRegion* cregion, int i_start) nogil:
    cdef int i
    cdef int j = i_start
    for i in range(i_start, cregion.pixels_max):
        if cregion.pixels[i] is not NULL:
            cregion.pixels[j] = cregion.pixels[i]
            j += 1
    for i in prange(j, cregion.pixels_max):
        cregion.pixels[i] = NULL
    cregion.pixels_istart = 0
    cregion.pixels_iend = cregion.pixels_n


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_remove_pixel_from_shells
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef inline void _cregion_shell_remove_gaps(cRegion* cregion, int i_shell, int i_start):
    cdef int i
    cdef int d = 0
    cdef int n_old = cregion.shell_n[i_shell]
    for i in range(i_start, n_old):
        if cregion.shells[i_shell][i] is NULL:
            d += 1
        cregion.shells[i_shell][i] = cregion.shells[i_shell][i + d]
        if i + 1 == n_old - d:
            break
    cregion.shell_n[i_shell] -= d
    for i in prange(n_old - d, n_old, nogil=True):
        cregion.shells[i_shell][i] = NULL


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_remove_pixel_from_shells_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef inline void _cregion_shell_remove_gaps_nogil(
    cRegion* cregion, int i_shell, int i_start,
) nogil:
    cdef int i
    cdef int n_old = cregion.shell_n[i_shell]
    cdef int d = 0
    for i in range(i_start, n_old):
        if cregion.shells[i_shell][i] is NULL:
            d += 1
        cregion.shells[i_shell][i] = cregion.shells[i_shell][i + d]
        if i + 1 == n_old - d:
            break
    cregion.shell_n[i_shell] -= d
    for i in prange(n_old - d, n_old):
        cregion.shells[i_shell][i] = NULL


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_remove_pixel_from_holes
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef inline void _cregion_hole_remove_gaps(cRegion* cregion, int i_hole, int i_start):
    cdef int i
    cdef int d=0
    cdef int n_old=cregion.hole_n[i_hole]
    for i in range(i_start, n_old):
        if cregion.holes[i_hole][i] is NULL:
            d += 1
        cregion.holes[i_hole][i] = cregion.holes[i_hole][i + d]
        if i + 1 == n_old - d:
            break
    cregion.hole_n[i_hole] -= d
    for i in prange(n_old - d, n_old, nogil=True):
        cregion.holes[i_hole][i] = NULL


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_remove_pixel_from_holes_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef inline void _cregion_hole_remove_gaps_nogil(
    cRegion* cregion, int i_hole, int i_start,
) nogil:
    cdef int i
    cdef int d=0
    cdef int n_old=cregion.hole_n[i_hole]
    for i in range(i_start, n_old):
        if cregion.holes[i_hole][i] is NULL:
            d += 1
        cregion.holes[i_hole][i] = cregion.holes[i_hole][i + d]
        if i + 1 == n_old - d:
            break
    cregion.hole_n[i_hole] -= d
    for i in prange(n_old - d, n_old):
        cregion.holes[i_hole][i] = NULL


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: > stormtrack::core::typedefs::_cregion_extend_shell
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_extend_shell
# :call: v stormtrack::core::typedefs::_cregion_reconnect_pixel
cdef void _cregion_insert_shell_pixel(cRegion* cregion, int i_shell, cPixel* cpixel):
    # print(
    #     f"< _cregion_insert_shell_pixel: ({cpixel.x}, {cpixel.y}) -> {cregion.id} "
    #     f"({cregion.shell_n}/{cregion.shell_max})"
    # )
    if i_shell > cregion.shells_n:
        raise Exception(
            f"error: _cregion_insert_shell_pixel: "
            f"i_shell={i_shell} > cregion.shells_n={cregion.shells_n}"
        )
    if cregion.shell_max[i_shell] == 0:
        _cregion_extend_shell(cregion, i_shell)
    cdef int pixel_i = cregion.shell_n[i_shell]
    cregion.shells[i_shell][pixel_i] = cpixel
    cregion.shell_n[i_shell] += 1
    if cregion.shell_n[i_shell] == cregion.shell_max[i_shell]:
        _cregion_extend_shell(cregion, i_shell)

    # If pixel not connected to region, double-check
    # that it belongs to the region and reconnect it
    # Turn on warning given that this should not happen anymore
    if cpixel.region is NULL or cpixel.region.id != cregion.id:
        _cregion_reconnect_pixel(cregion, cpixel, warn=True)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: > stormtrack::core::typedefs::_cregion_extend_hole
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_extend_hole
# :call: v stormtrack::core::typedefs::_cregion_reconnect_pixel
cdef void _cregion_insert_hole_pixel(cRegion* cregion, int i_hole, cPixel* cpixel):
    # print(
    #     f"< _cregion_insert_hole_pixel({i_hole}):
    #     f"{cregion.hole_n[i_hole]}/{cregion.hole_max[i_hole]}"
    # )
    if i_hole > cregion.holes_n:
        raise Exception(
            f"_cregion_insert_hole_pixel: "
            f"i_hole={i_hole} > cregion.holes_n={cregion.holes_n}"
        )
    cdef int pixel_i = cregion.hole_n[i_hole]
    cregion.holes[i_hole][pixel_i] = cpixel
    cregion.hole_n[i_hole] += 1
    if cregion.hole_n[i_hole] == cregion.hole_max[i_hole]:
        _cregion_extend_hole(cregion, i_hole)

    # If pixel not connected to region, double-check
    # that it belongs to the region and reconnect it
    # Turn on warning given that this should not happen anymore
    if cpixel.region is NULL or cpixel.region.id != cregion.id:
        _cregion_reconnect_pixel(cregion, cpixel, warn=True)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_insert_hole_pixel
# :call: > stormtrack::core::typedefs::_cregion_insert_shell_pixel
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cpixel_set_region
cdef void _cregion_reconnect_pixel(
    cRegion* cregion, cPixel* cpixel, bint warn,
):
    cdef int i
    for i in range(cregion.pixels_max):
        if cregion.pixels[i] is NULL:
            continue
        if cregion.pixels[i].x == cpixel.x and cregion.pixels[i].y == cpixel.y:
            break
    else:
        log.error(
            f"[_cregion_reconnect_pixel] shell pixel ({cpixel.x}, {cpixel.y}) "
            f"of region {cregion.id} not part of region"
        )
        exit(99)
    if warn:
        log.warning(
            f"[_cregion_reconnect_pixel] reconnect shell pixel "
            f"({cpixel.x}, {cpixel.y}) to region {cregion.id}"
        )
    cpixel_set_region(cpixel, cregion)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_insert_pixel
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cregion_pixels_remove_gaps
cdef void _cregion_extend_pixels(cRegion* cregion):
    cdef int i
    cdef int factor = 2
    cdef int nmax_old = cregion.pixels_max
    cdef int nmax_new = nmax_old * factor
    if nmax_old == 0:
        nmax_new = factor
    cdef cPixel** tmp = <cPixel**>malloc(nmax_old*sizeof(cPixel*))

    # First remove gaps and check whether this is already sufficient
    cdef float gap_threshold = 0.8 # rather arbitrary
    cdef int i_start = 0
    cregion_pixels_remove_gaps(cregion, i_start)
    if cregion.pixels_n < gap_threshold*cregion.pixels_max:
        return

    # Grow the array
    for i in range(nmax_old):
        tmp[i] = cregion.pixels[i]
    free(cregion.pixels)
    cregion.pixels = <cPixel**>malloc(nmax_new*sizeof(cPixel*))
    for i in range(nmax_old):
        cregion.pixels[i] = tmp[i]
    for i in prange(nmax_old, nmax_new, nogil=True):
        cregion.pixels[i] = NULL
    free(tmp)
    cregion.pixels_max = nmax_new


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_insert_pixel_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cregion_pixels_remove_gaps_nogil
cdef void _cregion_extend_pixels_nogil(cRegion* cregion) nogil:
    cdef int i
    cdef int factor = 2
    cdef int nmax_old = cregion.pixels_max
    cdef int nmax_new = nmax_old * factor
    if nmax_old == 0:
        nmax_new = factor
    cdef cPixel** tmp = <cPixel**>malloc(nmax_old*sizeof(cPixel*))

    # First remove gaps and check whether this is already sufficient
    cdef float gap_threshold = 0.8 # rather arbitrary
    cdef int i_start = 0
    cregion_pixels_remove_gaps_nogil(cregion, i_start)
    if cregion.pixels_n < gap_threshold*cregion.pixels_max:
        return

    # Grow the array
    for i in range(nmax_old):
        tmp[i] = cregion.pixels[i]
    free(cregion.pixels)
    cregion.pixels = <cPixel**>malloc(nmax_new*sizeof(cPixel*))
    for i in range(nmax_old):
        cregion.pixels[i] = tmp[i]
    for i in prange(nmax_old, nmax_new):
        cregion.pixels[i] = NULL
    free(tmp)
    cregion.pixels_max = nmax_new


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_insert_pixel_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef void _cregion_extend_shell(cRegion* cregion, int i_shell):
    if i_shell > cregion.shells_n:
        raise Exception(
            f"error: _cregion_insert_shell_pixel: i_shell={i_shell} > "
            f"cregion.shells_n={cregion.shells_n}"
        )
    cdef int i
    cdef int nmax_old = cregion.shell_max[i_shell]
    cdef int nmax_new = nmax_old * 5
    if nmax_old == 0:
        nmax_new = 5
    # print(f"< _cregion_extend_shell({i_shell}) [{cregion.id}]: {nmax_old} -> {nmax_new}")
    cdef cPixel** tmp = <cPixel**>malloc(nmax_old*sizeof(cPixel*))
    for i in range(nmax_old):
        tmp[i] = cregion.shells[i_shell][i]
    free(cregion.shells[i_shell])
    cregion.shells[i_shell] = <cPixel**>malloc(nmax_new*sizeof(cPixel*))
    for i in range(nmax_old):
        cregion.shells[i_shell][i] = tmp[i]
    for i in range(nmax_old, nmax_new):
        cregion.shells[i_shell][i] = NULL
    cregion.shell_max[i_shell] = nmax_new
    free(tmp)
    # print("< _cregion_extend_shell")


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_insert_hole_pixel
# :call: > stormtrack::core::typedefs::_cregion_new_hole
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef void _cregion_extend_hole(cRegion* cregion, int i_hole):
    if i_hole > cregion.holes_n:
        raise Exception(
            f"_cregion_insert_hole_pixel: "
            f"i_hole={i_hole} > cregion.holes_n={cregion.holes_n}"
        )
    cdef int i
    cdef int nmax_old = cregion.hole_max[i_hole]
    cdef int nmax_new = nmax_old * 5
    if nmax_old == 0:
        nmax_new = 5
    # print(f"< _cregion_extend_hole({i_hole}) [{cregion.id}]: {nmax_old} -> {nmax_new}")
    cdef cPixel** tmp = <cPixel**>malloc(nmax_old*sizeof(cPixel*))
    for i in range(nmax_old):
        tmp[i] = cregion.holes[i_hole][i]
    free(cregion.holes[i_hole])
    cregion.holes[i_hole] = <cPixel**>malloc(nmax_new*sizeof(cPixel*))
    for i in range(nmax_old):
        cregion.holes[i_hole][i] = tmp[i]
    cregion.hole_max[i_hole] = nmax_new
    free(tmp)
    # print("< _cregion_extend_hole")


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_new_shell
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef void _cregion_extend_shells(cRegion* cregion):
    cdef int i
    cdef int nmax_old = cregion.shells_max
    cdef int nmax_new = nmax_old * 5
    if nmax_old == 0:
        nmax_new = 5
    # print(f"< _cregion_extend_shells [{cregion.id}]: {nmax_old} -> {nmax_new}")
    cdef cPixel*** tmp_shells = <cPixel***>malloc(nmax_old*sizeof(cPixel**))
    cdef int* tmp_n=<int*>malloc(nmax_old*sizeof(int))
    cdef int* tmp_max=<int*>malloc(nmax_old*sizeof(int))
    for i in range(nmax_old):
        tmp_shells[i] = cregion.shells[i]
        tmp_n[i] = cregion.shell_n[i]
        tmp_max[i] = cregion.shell_max[i]
    free(cregion.shells)
    free(cregion.shell_n)
    free(cregion.shell_max)
    cregion.shells = <cPixel***>malloc(nmax_new*sizeof(cPixel**))
    cregion.shell_n = <int*>malloc(nmax_new*sizeof(int))
    cregion.shell_max = <int*>malloc(nmax_new*sizeof(int))
    for i in range(nmax_old):
        cregion.shells[i] = tmp_shells[i]
        cregion.shell_n[i] = tmp_n[i]
        cregion.shell_max[i] = tmp_max[i]
    for i in range(nmax_old, nmax_new):
        cregion.shells[i] = NULL
        cregion.shell_n[i] = 0
        cregion.shell_max[i] = 0
    cregion.shells_max = nmax_new
    free(tmp_shells)
    free(tmp_n)
    free(tmp_max)
    # print("< _cregion_extend_shells")


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_new_hole
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef void _cregion_extend_holes(cRegion* cregion):
    cdef int i
    cdef int nmax_old = cregion.holes_max
    cdef int nmax_new = nmax_old * 5
    if nmax_old == 0:
        nmax_new = 5
    # print(f"< _cregion_extend_holes [{cregion.id}]: {nmax_old} -> {nmax_new}")
    cdef cPixel*** tmp_holes = <cPixel***>malloc(nmax_old*sizeof(cPixel**))
    cdef int* tmp_n=<int*>malloc(nmax_old*sizeof(int))
    cdef int* tmp_max=<int*>malloc(nmax_old*sizeof(int))
    for i in range(nmax_old):
        tmp_holes[i] = cregion.holes[i]
        tmp_n[i] = cregion.hole_n[i]
        tmp_max[i] = cregion.hole_max[i]
    free(cregion.holes)
    free(cregion.hole_n)
    free(cregion.hole_max)
    cregion.holes = <cPixel***>malloc(nmax_new*sizeof(cPixel**))
    cregion.hole_n = <int*>malloc(nmax_new*sizeof(int))
    cregion.hole_max = <int*>malloc(nmax_new*sizeof(int))
    for i in range(nmax_old):
        cregion.holes[i] = tmp_holes[i]
        cregion.hole_n[i] = tmp_n[i]
        cregion.hole_max[i] = tmp_max[i]
    for i in range(nmax_old, nmax_new):
        cregion.holes[i] = NULL
        cregion.hole_n[i] = 0
        cregion.hole_max[i] = 0
    cregion.holes_max = nmax_new
    free(tmp_holes)
    free(tmp_n)
    free(tmp_max)
    # print("< _cregion_extend_holes")


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_extend_shell
# :call: v stormtrack::core::typedefs::_cregion_extend_shells
cdef void _cregion_new_shell(cRegion* cregion):
    # print(f"< _cregion_new_shell {cregion.shells_n}/{cregion.shells_max}")
    if cregion.shells_max == 0:
        _cregion_extend_shells(cregion)
    if cregion.shell_max[cregion.shells_n] == 0:
        _cregion_extend_shell(cregion, cregion.shells_n)
    cregion.shells_n += 1
    if cregion.shells_n == cregion.shells_max:
        _cregion_extend_shells(cregion)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_extend_hole
# :call: v stormtrack::core::typedefs::_cregion_extend_holes
cdef void _cregion_new_hole(cRegion* cregion):
    # print(f"< _cregion_new_hole {cregion.holes_n}/{cregion.holes_max}")
    if cregion.holes_max == 0:
        _cregion_extend_holes(cregion)
    if cregion.hole_max[cregion.holes_n] == 0:
        _cregion_extend_hole(cregion, cregion.holes_n)
    cregion.holes_n += 1
    if cregion.holes_n == cregion.holes_max:
        _cregion_extend_holes(cregion)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_connect
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_extend_connected
cdef void _cregion_add_connected(cRegion* cregion, cRegion* cregion_other):
    # print(f"< _cregion_add_connected {cregion.id} <- {cregion_other.id} (no. {cregion.connected_n})")
    cdef int i
    for i in range(cregion.connected_n):
        if cregion.connected[i].id == cregion_other.id:
            # print("> _cregion_add_connected (already present)")
            return
    # print(f"add connected region: {cregion.id} <- {cregion_other.id} ({cregion.connected_n})")
    if cregion.connected_n == 0:
        _cregion_extend_connected(cregion)
    cregion.connected[cregion.connected_n] = cregion_other
    cregion.connected_n += 1
    if cregion.connected_n == cregion.connected_max:
        _cregion_extend_connected(cregion)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_add_connected
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef void _cregion_extend_connected(cRegion* cregion):
    cdef bint debug = False
    cdef int i
    cdef int nmax_old = cregion.connected_max
    cdef int nmax_new = nmax_old * 5
    if nmax_old == 0:
        nmax_new = 5
    # print(f"< _cregion_extend_connected {cregion.id}: {nmax_old} -> {nmax_new}")
    cdef cRegion** tmp = <cRegion**>malloc(nmax_old*sizeof(cRegion*))
    for i in range(nmax_old):
        tmp[i] = cregion.connected[i]
    free(cregion.connected)
    cregion.connected = <cRegion**>malloc(nmax_new*sizeof(cRegion*))
    for i in range(nmax_old):
        cregion.connected[i] = tmp[i]
    for i in range(nmax_old, nmax_new):
        cregion.connected[i] = NULL
    cregion.connected_max = nmax_new
    free(tmp)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::typedefs::cregion_cleanup
# :call: > stormtrack::core::typedefs::cregions_reset
# :call: > stormtrack::core::typedefs::cregions_store_reset
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cpixel_unlink_region
# :call: v stormtrack::core::typedefs::_cregion_reset_connected
cdef void cregion_reset(cRegion* cregion, bint unlink_pixels, bint reset_connected):
    cdef bint debug = False
    if debug:
        log.debug(f"< cregion_reset {cregion.id}")
    if cregion is NULL:
        return
    cdef int i
    cdef int j
    cdef int k
    cdef cRegion* other_cregion

    # Connected regions
    if debug:
        log.debug(f" -> connected {cregion.connected_max}")
    if reset_connected and cregion.connected_n > 0:
        _cregion_reset_connected(cregion, unlink=True)

    # Pixels
    if debug:
        log.debug(f" -> pixels {cregion.pixels_max}")
    cdef cPixel* cpixel
    if cregion.pixels_n > 0:
        # Unlink connected pixels
        for i in prange(cregion.pixels_istart, cregion.pixels_iend, nogil=True):
            if unlink_pixels:
                _cpixel_unlink_region(cregion.pixels[i], cregion)
            cregion.pixels[i] = NULL
    cregion.pixels_n = 0
    cregion.pixels_istart = 0
    cregion.pixels_iend = 0

    # Shells pixels
    if debug:
        log.debug(f" -> shells {cregion.shells_max}")
    for i in prange(cregion.shells_n, nogil=True):
        for j in range(cregion.shell_n[i]):
            cregion.shells[i][j] = NULL
        cregion.shell_n[i] = 0
    cregion.shells_n = 0

    # Holes pixels
    if debug:
        log.debug(f" -> holes {cregion.holes_max}")
    for i in prange(cregion.holes_n, nogil=True):
        for j in range(cregion.hole_n[i]):
            cregion.holes[i][j] = NULL
        cregion.hole_n[i] = 0
    cregion.holes_n = 0


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_reset
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cpixel_set_region
cdef inline void _cpixel_unlink_region(cPixel* cpixel, cRegion* cregion) nogil:
    if (
        cpixel is not NULL
        and cpixel.region is not NULL
        and cpixel.region.id == cregion.id
    ):
        # if debug:
        #     log.debug(
        #         f"cleanup region {cregion.id} - disconnect pixel ({cpixel[i].x}, "
        #         f"{cpixel.y})"
        #     )
        cpixel_set_region(cpixel, NULL)
        cpixel.is_feature_boundary = False


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_reset
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cregion_remove_connected
cdef inline void _cregion_reset_connected(cRegion* cregion, bint unlink):
    cdef int i
    for i in range(cregion.connected_n):
        if unlink:
            other_cregion = cregion.connected[i]
            cregion_remove_connected(other_cregion, cregion)
        cregion.connected[i] = NULL
    cregion.connected_n = 0


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_reset_connected
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef void cregion_remove_connected(cRegion* cregion, cRegion* cregion_other):
    cdef int i
    cdef int j
    for i in range(cregion.connected_n):
        if cregion.connected[i].id == cregion.id:
            break
    else:
        return
    cregion.connected_n -= 1
    for j in range(i, cregion.connected_n):
        cregion.connected[j] = cregion.connected[j+1]
    cregion.connected[cregion.connected_n] = NULL


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::Feature::cleanup_cregion
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::eliminate_regions_by_size
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::cregion_merge
# :call: > stormtrack::core::typedefs::cregions_cleanup
# :call: > stormtrack::core::typedefs::cregions_link_region
# :call: > stormtrack::core::typedefs::cregions_store_cleanup
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cregion_reset
cdef void cregion_cleanup(cRegion* cregion, bint unlink_pixels, bint reset_connected):
    cdef bint debug = False
    if debug:
        log.debug(f"< cregion_cleanup {cregion.id}")
    cdef int i

    if debug:
        log.debug(f" -> reset cregion {cregion.id}")
    cregion_reset(cregion, unlink_pixels, reset_connected)

    if debug:
        log.debug(" -> clean up connected")
    if cregion.connected_max > 0:
        free(cregion.connected)
        cregion.connected = NULL
        cregion.connected_max = 0

    if debug:
        log.debug(" -> clean up pixels")
    if cregion.pixels_max > 0:
        free(cregion.pixels)
        cregion.pixels = NULL
        cregion.pixels_max = 0

    if debug:
        log.debug(" -> clean up shell")
    if cregion.shells_max > 0:
        for i in range(cregion.shells_max):
            if cregion.shell_max[i] > 0:
                free(cregion.shells[i])
        free(cregion.shells)
        free(cregion.shell_n)
        free(cregion.shell_max)
        cregion.shells = NULL
        cregion.shell_n = NULL
        cregion.shell_max = NULL
        cregion.shells_n = 0
        cregion.shells_max = 0

    if debug:
        log.debug(" -> clean up holes")
    if cregion.holes_max > 0:
        for i in range(cregion.holes_max):
            if cregion.hole_max[i] > 0:
                free(cregion.holes[i])
        free(cregion.holes)
        free(cregion.hole_n)
        free(cregion.hole_max)
        cregion.holes = NULL
        cregion.hole_n = NULL
        cregion.hole_max = NULL
        cregion.holes_n = 0
        cregion.holes_max = 0

    if debug:
        log.debug("> cregion_cleanup")


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::find_existing_region
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cregion_cleanup
# :call: v stormtrack::core::typedefs::cregion_insert_pixel
cdef cRegion* cregion_merge(cRegion* cregion1, cRegion* cregion2):
    cdef bint debug = False
    cdef int i
    if debug:
        log.debug(f"< cregion_merge {cregion2.id} -> {cregion1.id}")
    for i in range(cregion2.pixels_istart, cregion2.pixels_iend):
        if cregion2.pixels[i] is not NULL:
            cregion_insert_pixel(
                cregion1, cregion2.pixels[i], link_region = True, unlink_pixel = False,
            )
    # SR_TODO reset_connected necessary?
    cregion_cleanup(cregion2, unlink_pixels = False, reset_connected = True)
    return cregion1


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::cregions_create
# :call: v stormtrack::core::typedefs::cregions_determine_boundaries
cdef void cregion_determine_boundaries(cRegion* cregion, cGrid* grid) except *:
    cdef cRegions cregions = cregions_create(1)
    cregions.n = 1
    cregions.regions[0] = cregion
    cregions_determine_boundaries(&cregions, grid)
    free(cregions.regions)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::typedefs::cregion_determine_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: v stormtrack::core::typedefs::cregion_reset_boundaries
cdef void cregions_determine_boundaries(cRegions* cregions, cGrid* grid) except *:
    cdef bint debug = False
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
        _cregion_determine_boundaries_core(cregion, grid)
    if n_empty > 0:
        log.warning(f"{n_empty}/{cregions.n} regions empty")


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_determine_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::_cregion_insert_hole_pixel
# :call: v stormtrack::core::typedefs::_cregion_insert_shell_pixel
# :call: v stormtrack::core::typedefs::_cregion_new_hole
# :call: v stormtrack::core::typedefs::_cregion_new_shell
# :call: v stormtrack::core::typedefs::_determine_boundary_pixels_raw
# :call: v stormtrack::core::typedefs::_reconstruct_boundaries
# :call: v stormtrack::core::typedefs::categorize_boundaries
# :call: v stormtrack::core::typedefs::cpixel_set_region
# :call: v stormtrack::core::typedefs::cregion_cleanup
# :call: v stormtrack::core::typedefs::cregions_cleanup
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

    # - # SR_TMP <
    # - k = 0
    # - for i in range(cboundaries.n):
    # -     if boundary_is_shell[i]:
    # -         k += 1
    # - if k > 1:
    # -     raise NotImplementedError(
    # -         "cregions_find_connected for multiple shells per feature"
    # -     )
    # - # SR_TMP >

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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::structs::get_matching_neighbor_id
# :call: v stormtrack::core::tables::neighbor_link_stat_table_init
# :call: v stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: v stormtrack::core::typedefs::_extract_closed_path
# :call: v stormtrack::core::typedefs::_find_link_to_continue
# :call: v stormtrack::core::typedefs::cregion_cleanup
# :call: v stormtrack::core::typedefs::cregion_insert_pixel
# :call: v stormtrack::core::typedefs::cregion_northernmost_pixel
# :call: v stormtrack::core::typedefs::cregions_create
# :call: v stormtrack::core::typedefs::cregions_link_region
# :call: v stormtrack::core::typedefs::grid_new_region
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
            current_boundary = grid_new_region(grid)
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
                    with cdivision(True):
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef bint _find_link_to_continue(
    cPixel** cpixel,
    np.uint8_t *i_neighbor,
    cRegion* boundary_pixels,
    np.int8_t ***neighbor_link_stat_table,
):
    """At start of iteration, find pixel and neighbor and check if done."""
    cdef bint debug = False
    if debug:
        log.debug(f"< _find_link_to_continue (iter {iter})")
    cdef int i
    cdef int j
    cdef int k
    cdef int l
    cdef bint done = True

    # Loop over all boundary pixels
    for i in range(boundary_pixels.pixels_istart, boundary_pixels.pixels_iend):
        cpixel[0] = boundary_pixels.pixels[i]
        if cpixel[0] is NULL:
            continue

        # Select neighbor to continue with (if any are left)
        for j in range(cpixel[0].neighbors_max):
            if cpixel[0].neighbors[j] is not NULL:
                if neighbor_link_stat_table[cpixel[0].x][cpixel[0].y][j] > 0:

                    # Pixel found with which to continue! Not yet done!
                    i_neighbor[0] = j
                    done = False
                    break

        # Remove pixel (don't use cregion_remove_pixel; too slow)
        boundary_pixels.pixels[i] = NULL
        boundary_pixels.pixels_n -= 1

        # DBG_PERMANENT <
        if boundary_pixels.pixels_n < 0:
            log.error("_find_link_to_continue: boundary_pixels.pixels_n < 0 !!!")
            exit(4)
        # DBG_PERMANENT >

        if not done:
            break

    if debug:
        log.debug("> _find_link_to_continue (done={done})")
    return done


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::boundary_must_be_a_shell
# :call: v stormtrack::core::typedefs::categorize_boundary_is_shell
# :call: v stormtrack::core::typedefs::cregion_check_validity
# :call: v stormtrack::core::typedefs::cregions_find_northernmost_uncategorized_region
# :call: v stormtrack::core::typedefs::neighbor_pixel_angle
# :call: v stormtrack::core::typedefs::sign
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
            angle = neighbor_pixel_angle(cpixel, cpixel_pre)
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::categorize_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
cdef int neighbor_pixel_angle(
    cPixel* cpixel1, cPixel* cpixel2, bint minus=True,
) except -1:

    cdef int dx = cpixel1.x - cpixel2.x
    cdef int dy = cpixel1.y - cpixel2.y
    cdef int angle

    if dx == 1 and dy == 0:
        angle = 0
    elif dx == 1 and dy == 1:
        angle = 45
    elif dx == 0 and dy == 1:
        angle = 90
    elif dx == -1 and dy == 1:
        angle = 135
    elif dx == -1 and dy == 0:
        angle = 180
    elif dx == -1 and dy == -1:
        angle = 225
    elif dx == 0 and dy == -1:
        angle = 270
    elif dx == 1 and dy == -1:
        angle = 315
    else:
        raise Exception(
            f"cannot derive angle between ({cpixel1.x}, {cpixel1.y}) and "
            f"({cpixel2.x}, {cpixel2.y})"
        )

    if minus and angle > 180:
        angle -= 360

    return angle


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef bint _extract_closed_path(cRegion* boundary):
    """Extract the longest closed segment (return value indicates success).

    Method:
    - Find all pixels that occur at least twice in the boundary.
    - For each, count the number of pixels towards both ends.
    - Chose the pixel for which this number is smallest.
    - Cut off all pixels farther towards one end than that pixel.

    """
    # Note: This might be pretty slow with bigger features!
    # The problem is that only very few pixels occur multiple times,
    # but the whole array is checked for each pixel nontheless.
    # What we assume, though, is that the pixels that occur multiple
    # times and we're interested in are located near both ends of the
    # array. This information can be leveraged by an algorithm that
    # works somewhat as follows.
    #
    # - Determine a certain "chunk size" (e.g. 20 pixels; tuning parameter)
    # - Start in forward direction.
    # - For each pixel in the first "n_chunk_size" elements, check all pixels
    #   in the last "n_chunk_size" elements for equality.
    # - If we find pixels that occur multiple times, we select the one
    #   with the smallest combined distance to start and end, and we're done.
    #
    # - Otherwise, we switch direction.
    # - We double the size of the "leading segment" (now the one from the end
    #   of the array) and shift the other segment (the one from the beginning)
    #   by one segment size (i.e. by "n_chunk_size").
    # - For every pixel in the leading segment (size 2*n_chunk_size), we
    #   check every pixel in the other segment (size n_chunk_size).
    # - Again, we're done if we find pixels occurring multiple times.
    #
    # - Now I'm not yet exactly sure how to continue such that all segments
    #   are checked against all, but that can be worked out in case it is
    #   necessary to implement. For now, just leave the idea here.
    #
    # print("< _extract_closed_path")
    cdef bint debug = False
    cdef int i
    cdef int j
    cdef int k
    cdef int l
    cdef int ind0
    cdef int ind1
    cdef int dist
    cdef int dist_min = -1

    # DBG_BLOCK <
    if debug:
        log.debug("<<<<")
        for i in range(boundary.pixels_max):
            if boundary.pixels[i] is not NULL:
                log.debug(
                    f"[{boundary.id}] {i} "
                    f"({boundary.pixels[i].x}, {boundary.pixels[i].y})"
                )
        log.debug("----")
    # DBG_BLOCK >

    # Determine pair of identical pixels with smallest combined
    # respective distance to the beginning and end of the array
    for i in range(boundary.pixels_max):
        if boundary.pixels[i] is NULL:
            continue
        for j in range(boundary.pixels_max-1, i+1, -1):
            if boundary.pixels[j] is NULL:
                continue
            if (boundary.pixels[i].x == boundary.pixels[j].x and
                    boundary.pixels[i].y == boundary.pixels[j].y):
                dist = 1
                for k in range(i+1, j):
                    if boundary.pixels[k] is not NULL:
                        dist += 1
                if dist_min < 0 or dist < dist_min:
                    dist_min = dist
                    ind0 = i
                    ind1 = j
                    if debug:
                        log.debug(f"ind0 = {i}")
                        log.debug(f"ind1 = {j}")
                break

    # Return if none has been found
    if dist_min < 0:
        if debug:
            log.debug("> _extract_closed_path: False")
        return False

    # Trim the array to the longest closed segment
    cdef int nold = boundary.pixels_n
    cdef int nnew = nold - dist
    cdef cPixel** tmp = <cPixel**>malloc(nnew*sizeof(cPixel*))

    k = 0
    for i in range(boundary.pixels_max):
        if boundary.pixels[i] is NULL:
            if debug:
                log.debug(f"{i}/{k} continue (NULL)")
            boundary.pixels[i] = NULL
            continue
        if i < ind0:
            if debug:
                log.debug(f"{i}/{k} continue ({i} < {ind0})")
            boundary.pixels[i] = NULL
            continue
        if i > ind1:
            if debug:
                log.debug(f"{i}/{k} boundary.pixels[{i}] = NULL ({i} > {ind1})")
            boundary.pixels[i] = NULL
            continue
        if i > k:
            if debug:
                log.debug(f"{i}/{k} boundary.pixels[{k}] = boundary.pixels[{i}]")
                log.debug(f"{i}/{k} boundary.pixels[{i}] = NULL")
            boundary.pixels[k] = boundary.pixels[i]
            boundary.pixels[i] = NULL
        k += 1
    if debug:
        log.debug(f"boundary({boundary.id}).pixels_n = {k}")
    boundary.pixels_n = k
    boundary.pixels_istart = 0
    boundary.pixels_iend = k

    # DBG_BLOCK <
    if debug:
        log.debug("----")
        for i in range(boundary.pixels_max):
            if boundary.pixels[i] is not NULL:
                log.debug(
                    f"[{boundary.id}] {i} "
                    f"({boundary.pixels[i].x}, {boundary.pixels[i].y})"
                )
        log.debug(">>>>")
    # DBG_BLOCK >

    if debug:
        log.debug("> _extract_closed_path: True")
        exit(2)
    return True


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::categorize_boundaries
# :call: v --- CALLING ---
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::categorize_boundaries
# :call: v --- CALLING ---
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::categorize_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef void cregion_check_validity(cRegion* cregion, int idx) except *:
    """Check the validity of a cregion."""
    cdef cPixel* cpixel = cregion.pixels[0]
    cdef cPixel* cpixel_pre = cregion.pixels[cregion.pixels_n - 1]
    if not cpixel.x == cpixel_pre.x and cpixel.y == cpixel_pre.y:
        raise Exception(
            f"cregion # {idx}: first and last ({cregion.pixels_n - 1}th) pixel differ: "
            f"({cpixel.x}, {cpixel.y}) != ({cpixel_pre.x}, {cpixel_pre.y})"
        )


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::cregions_find_northernmost_uncategorized_region
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef cPixel* cregion_northernmost_pixel(cRegion* cregion):
    """Find northwesternmost pixel (northmost has priority)."""
    cdef cPixel* selection = NULL
    cdef cPixel* cpixel
    cdef int i
    for i in range(cregion.pixels_max):
        cpixel = cregion.pixels[i]
        if cpixel is NULL:
            continue
        if selection is NULL:
            selection = cpixel
            continue
        if cpixel.y > selection.y:
            selection = cpixel
        elif cpixel.y == selection.y and cpixel.x < selection.x:
            selection = cpixel
    return selection


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::categorize_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::cregion_northernmost_pixel
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_determine_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
cdef void cregion_reset_boundaries(cRegion* cregion):
    cdef int i_pixel
    cdef int i_shell
    cdef int i_hole

    for i_pixel in range(cregion.pixels_max):
        if cregion.pixels[i_pixel] is not NULL:
            cregion.pixels[i_pixel].is_feature_boundary = False

    for i_shell in range(cregion.shells_n):
        for i_pixel in range(cregion.shell_max[i_shell]):
            cregion.shells[i_shell][i_pixel] = NULL
        cregion.shell_n[i_shell] = 0
    cregion.shells_n = 0

    for i_hole in range(cregion.holes_n):
        for i_pixel in range(cregion.hole_max[i_hole]):
            cregion.holes[i_hole][i_pixel] = NULL
        cregion.hole_n[i_hole] = 0
    cregion.holes_n = 0


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::typedefs::_collect_neighbors
# :call: v stormtrack::core::typedefs::cregion_get_unique_id
# :call: v stormtrack::core::typedefs::cregion_init
# :call: v stormtrack::core::typedefs::cregion_insert_pixel
cdef cRegion _determine_boundary_pixels_raw(cRegion* cregion, cGrid* grid):
    """Determine all boundary pixels of a feature, regardless which boundary.

    The neighbors are determined "by hand" because reverse connectivity needs
    to be used (i.e. 4-/8-connectivity neighbors for 8-/4-connectivity).

    The reason for this inversal is that for 4-connectivity, corner pixels
    which only have a diagonal non-feature neighbor can belong to the boundary
    because there are only right angles in the boundary (no diagonals).
    Thus all pixels with any 8-connectivity non-feature neighbors are boundary
    pixels. Conversely, for 8-connectivity, there are diagnoals in the
    boundary. Thus, only pixels with 4-connectivity non-feature neighbours are
    boundary pixels, whereas corner pixels with diagonal non-feature neighbors
    are "short-cut" by a diagonal boundary connection.
    """
    cdef bint debug = False
    if debug:
        log.debug(f"< _determine_boundary_pixels_raw [{cregion.id}]")
    cdef int i
    cdef int j
    cdef int n
    cdef cPixel* cpixel
    cdef int inverse_connectivity
    cdef int n_neighbors
    cdef int n_neighbors_feature
    cdef int n_neighbors_domain_boundary
    cdef int n_neighbors_same_feature
    cdef int n_neighbors_other_feature
    cdef int n_neighbors_background
    cdef cPixel** neighbors = <cPixel**>malloc(8*sizeof(cPixel*))
    cdef cRegion boundary_pixels
    cregion_init(&boundary_pixels, cregion_conf_default(), cregion_get_unique_id())

    # DBG_BLOCK <
    if debug:
        log.debug(f"\n[{cregion.id}] {cregion.pixels_n} pixels:")
        j = 0
        for i in range(cregion.pixels_max):
            if cregion.pixels[i] is not NULL:
                log.debug(f"{j} ({cregion.pixels[i].x}, {cregion.pixels[i].y})")
                j += 1
        log.debug("")
    # DBG_BLOCK >

    # Determine neighbor connectivity (inverse)
    if grid.constants.connectivity == 4:
        inverse_connectivity = 8
    elif grid.constants.connectivity == 8:
        inverse_connectivity = 4
    else:
        raise NotImplementedError(f"connectivity={grid.constants.connectivity}")

    # Check all pixels of the region for non-feature neighbors
    n = cregion.pixels_n
    cdef cPixel** cpixels_tmp = <cPixel**>malloc(n*sizeof(cPixel*))
    j = 0
    for i in range(cregion.pixels_max):
        if cregion.pixels[i] is not NULL:
            cpixels_tmp[j] = cregion.pixels[i]
            j += 1
    for i in range(n):
        cpixel = cpixels_tmp[i]
        if cpixel is NULL:
            continue
        # SR_DBG_PERMANENT <
        if cpixel.region is NULL:
            log.error(f"cpixel({cpixel.x}, {cpixel.y}).region is NULL")
            exit(23)
        # SR_DBG_PERMANENT >

        # Collect all pixel neighbors with inverse connectivity
        n_neighbors = _collect_neighbors(
            cpixel.x,
            cpixel.y,
            neighbors,
            grid.pixels,
            &grid.constants,
            inverse_connectivity,
        )
        if debug:
            log.debug(
                f"[{cpixel.region.id}] ({cpixel.x}, {cpixel.y}): "
                f"({n_neighbors} neighbors)"
            )

        # If any neighbor belongs not to the region, it's a boundary pixel
        n_neighbors_same_feature = 0
        n_neighbors_other_feature = 0
        n_neighbors_background = 0
        n_neighbors_domain_boundary = 0
        for j in range(grid.constants.n_neighbors_max):
            if inverse_connectivity == 4 and j%2 > 0:
                continue
            if neighbors[j] is NULL:
                if debug:
                    log.debug(f" -> neighbor {j}: outside the domain")
                n_neighbors_domain_boundary += 1
            elif neighbors[j].region is NULL:
                if debug:
                    log.debug(
                        f" -> neighbor {j} ({neighbors[j].x}, {neighbors[j].y}): "
                        f"region is NULL"
                    )
                n_neighbors_background += 1
                continue
            elif neighbors[j].region.id == cpixel.region.id:
                if debug:
                    log.debug(
                        f" -> neighbor {j} ({neighbors[j].x}, {neighbors[j].y}): "
                        f"same region {cpixel.region.id}"
                    )
                n_neighbors_same_feature += 1
            else:
                if debug:
                    log.debug(
                        f" -> neighbor {j} ({neighbors[j].x}, {neighbors[j].y}): "
                        f"other region {neighbors[j].region.id}"
                    )
                n_neighbors_other_feature += 1
        # DBG_BLOCK <
        if debug:
            log.debug(f" -> {n_neighbors} neighbors:")
            log.debug(f"    {n_neighbors_same_feature} same feature {cpixel.region.id}")
            log.debug(f"    {n_neighbors_other_feature} other feature(s)")
            log.debug(f"    {n_neighbors_background} background")
            log.debug(f"    {n_neighbors_domain_boundary} domain boundary")
        # DBG_BLOCK >

        # If the pixel is at the domain boundary, it's a feature boundary pixel
        if n_neighbors_same_feature > 0 and n_neighbors_domain_boundary > 0:
            if debug:
                log.debug(" -> is at domain boundary")
            pass

        # If all neighbors belong to the same feature, it's not a boundary pixel
        elif n_neighbors_same_feature == n_neighbors:
            if debug:
                log.debug(f"     -> all same feature {cpixel.region.id} ({n_neighbors})")
                log.debug(f" => not feature boundary pixel")
            continue

        if debug:
            log.debug(f" => feature boundary pixel no. {boundary_pixels.pixels_n}")
        cpixel.is_feature_boundary = True
        cregion_insert_pixel(
            &boundary_pixels, cpixel, link_region=False, unlink_pixel=False,
        )
        if debug:
            log.debug(f"boundary[{boundary_pixels.pixels_n}] = ({cpixel.x}, {cpixel.y})")

    free(cpixels_tmp)
    free(neighbors)

    # print(f"> _determine_boundary_pixels_raw {cregion.id} {boundary_pixels_n[0]}")
    return boundary_pixels


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_determine_boundary_pixels_raw
# :call: > stormtrack::core::typedefs::grid_create_pixels
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::typedefs::_cpixel_get_neighbor
@profile(False)
@boundscheck(False)
@wraparound(False)
cdef inline int _collect_neighbors(
    np.int32_t i,
    np.int32_t j,
    cPixel** neighbors,
    cPixel** cpixels,
    cConstants* constants,
    int connectivity, # must be passed explicitly (not same as constants.~)
):
    cdef bint debug = False
    if debug:
        log.debug(
            f"< _collect_neighbors ({i}, {j}) {constants.nx}x{constants.ny} "
            f"({connectivity})"
        )
    cdef int index
    cdef int n_neighbors = 0
    for index in range(8):
        neighbors[index] = _cpixel_get_neighbor(
            &cpixels[i][j],
            index,
            cpixels,
            constants.nx,
            constants.ny,
            connectivity,
        )
        if neighbors[index] is not NULL:
            n_neighbors += 1
    return n_neighbors


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cpixel_count_neighbors_in_cregion
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::typedefs::_cpixel_get_neighbor
cdef cPixel* cpixel_get_neighbor(
    cPixel* cpixel,
    int index,
    cPixel** cpixels,
    np.int32_t nx,
    np.int32_t ny,
    int connectivity,
):
    return _cpixel_get_neighbor(cpixel, index, cpixels, nx, ny, connectivity)


# :call: > --- CALLERS ---
# :call: > typedefs:_collect_neighbors
# :call: > typedefs:cpixel_get_neighbor
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
@profile(False)
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef inline cPixel* _cpixel_get_neighbor(
    cPixel* cpixel,
    int index,
    cPixel** cpixels,
    np.int32_t nx,
    np.int32_t ny,
    int connectivity,
):
    cdef bint debug = False
    if debug:
        log.debug(f"< cpixel_get_neighbor ({cpixel.x}, {cpixel.y}) {index}")
    cdef np.int32_t x = cpixel.x
    cdef np.int32_t i = nx
    cdef np.int32_t y = cpixel.y
    cdef np.int32_t j = ny
    if connectivity == 8:
        pass
    elif connectivity == 4:
        if index % 2 > 0:
            return NULL
    else:
        log.error(f"cpixel_get_neighbor: invalid connectivity {connectivity}")
        exit(2)

    # North
    if index == 0 and y < ny-1:
        i = x
        j = y + 1

    # East
    if index == 2 and x < nx-1:
        i = x + 1
        j = y

    # South
    if index == 4 and y > 0:
        i = x
        j = y - 1

    # West
    if index == 6 and x > 0:
        i = x - 1
        j = y

    # North-east
    if index == 1 and x < nx-1 and y < ny-1:
        i = x + 1
        j = y + 1

    # South-east
    if index == 3 and x < nx-1 and y > 0:
        i = x + 1
        j = y - 1

    # South-west
    if index == 5 and x > 0 and y > 0:
        i = x - 1
        j = y - 1

    # North-west
    if index == 7 and x > 0 and y < ny-1:
        i = x - 1
        j = y + 1

    if i < nx and j < ny:
        if debug:
            log.debug(f"({x}, {y}) neighbors({index}) = ({i}, {j})")
        return &cpixels[i][j]
    if debug:
        log.debug(f"({x}, {y}) neighbors({index}) = NULL")
    return NULL


# :call: > --- CALLERS ---
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_overlap_core
cdef bint cregion_overlaps(cRegion* cregion, cRegion* cregion_other):
    """Check whether two cregions overlap cregions."""
    cdef int n = _cregion_overlap_core(
        cregion, cregion_other, table=NULL, table_other=NULL, count=False,
    )
    if n > 0:
        return True
    return False


# :call: > --- CALLERS ---
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::PixelRegionTable
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_overlap_core
cdef bint cregion_overlaps_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
):
    return _cregion_overlap_core(
        cregion, cregion_other, table, table_other, count=False,
    )


# :call: > --- CALLERS ---
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_overlap_core
cdef int cregion_overlap_n(cRegion* cregion, cRegion* cregion_other):
    """Count overlapping pixels between two cregions."""
    return _cregion_overlap_core(
        cregion, cregion_other, table=NULL, table_other=NULL, count=True,
    )


# :call: > --- CALLERS ---
# :call: > stormtrack::core::tracking::FeatureTracker::_compute_successor_probabilities
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::PixelRegionTable
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_overlap_core
cdef int cregion_overlap_n_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
):
    return _cregion_overlap_core(cregion, cregion_other, table, table_other, count=True)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregion_overlap_n
# :call: > stormtrack::core::typedefs::cregion_overlap_n_tables
# :call: > stormtrack::core::typedefs::cregion_overlaps
# :call: > stormtrack::core::typedefs::cregion_overlaps_tables
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::PixelRegionTable
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cregion_determine_bbox
cdef int _cregion_overlap_core(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
    bint count,
):
    cdef int i
    cdef int j
    cdef np.int32_t x
    cdef np.int32_t y
    cdef int n_overlap = 0

    # Determine bigger feature to use for outer loop
    cdef cRegion* bigger
    cdef cRegion* smaller
    cdef PixelRegionTable table_bigger
    cdef PixelRegionTable table_smaller
    if cregion.pixels_n >= cregion_other.pixels_n:
        bigger = cregion
        smaller = cregion_other
        table_bigger = table
        table_smaller = table_other
    else:
        bigger = cregion_other
        smaller = cregion
        table_bigger = table_other
        table_smaller = table

    # Use lookup tables if given (Note: actually used is only the table of
    # the bigger features, yet it is necessary to always pass both tables)
    if table is not NULL and table_other is not NULL:
        for i in range(smaller.pixels_istart, smaller.pixels_iend):
            if smaller.pixels[i] is NULL:
                continue
            x = smaller.pixels[i].x
            y = smaller.pixels[i].y
            for j in range(table_bigger[x][y].n):
                if table_bigger[x][y].slots[j].region.id == bigger.id:
                    if not count:
                        return True
                    n_overlap += 1
                    break
        if not count:
            return False
        return n_overlap

    # Determine bounding boxes
    cdef cPixel ll_bigger
    cdef cPixel ur_bigger
    cdef cPixel ll_smaller
    cdef cPixel ur_smaller
    cregion_determine_bbox(bigger, &ll_bigger, &ur_bigger)
    cregion_determine_bbox(smaller, &ll_smaller, &ur_smaller)

    # Check bounding boxes for overlap
    if (
        ur_bigger.x < ll_smaller.x
        or ur_bigger.y < ll_smaller.y
        or ll_bigger.x > ur_smaller.x
        or ll_bigger.y > ur_smaller.y
    ):
        return 0

    # Compare all pixels
    cdef cPixel* px_bigger
    cdef cPixel* px_smaller
    # Compare actual pixels
    for i in range(bigger.pixels_istart, bigger.pixels_iend):
        px_bigger = bigger.pixels[i]
        if px_bigger is NULL:
            continue
        # Check if pixel is inside the other pixel's bounding box
        if (
            px_bigger.x < ll_smaller.x
            or px_bigger.x > ur_smaller.x
            or px_bigger.y < ll_smaller.y
            or px_bigger.y > ur_smaller.y
        ):
            continue
        # Check all pixels
        for j in range(smaller.pixels_istart, smaller.pixels_iend):
            px_smaller = smaller.pixels[j]
            if px_smaller is NULL:
                continue
            if px_bigger.x == px_smaller.x and px_bigger.y == px_smaller.y:
                if count:
                    n_overlap += 1
                else:
                    return 1
                break

    return n_overlap


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
@boundscheck(False)
@wraparound(False)
cdef int cregion_overlap_n_mask(cRegion* cregion, np.ndarray[np.uint8_t, ndim=2] mask):
    cdef int i
    cdef int n=0
    cdef cPixel* cpixel
    for i in range(cregion.pixels_istart, cregion.pixels_iend):
        cpixel = cregion.pixels[i]
        if cpixel is not NULL:
            if mask[cpixel.x, cpixel.y] > 0:
                n += 1
    return n


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_overlap_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef void cregion_determine_bbox(
    cRegion* cregion, cPixel* lower_left, cPixel* upper_right,
):
    cdef int i_pixel
    cdef cPixel* cpixel
    # Initialize to first pixel
    for i_pixel in range(cregion.pixels_istart, cregion.pixels_iend):
        cpixel = cregion.pixels[i_pixel]
        if cpixel is not NULL:
            lower_left.x = cpixel.x
            lower_left.y = cpixel.y
            upper_right.x = cpixel.x
            upper_right.y = cpixel.y
            break
    # Run over all pixels
    for i_pixel in range(cregion.pixels_istart, cregion.pixels_iend):
        cpixel = cregion.pixels[i_pixel]
        if cpixel is not NULL:
            # lower_left.x = min(lower_left.x, cpixel.x)
            # lower_left.y = min(lower_left.y, cpixel.y)
            # upper_right.x = max(upper_right.x, cpixel.x)
            # upper_right.y = max(upper_right.y, cpixel.y)
            if cpixel.x < lower_left.x:
                lower_left.x = cpixel.x
            if cpixel.y < lower_left.y:
                lower_left.y = cpixel.y
            if cpixel.x > upper_right.x:
                upper_right.x = cpixel.x
            if cpixel.y > upper_right.y:
                upper_right.y = cpixel.y


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_create
# :call: v --- CALLING ---
cdef np.uint64_t cregions_get_unique_id():
    global CREGIONS_NEXT_ID
    cdef np.uint64_t rid = CREGIONS_NEXT_ID
    CREGIONS_NEXT_ID += 1
    return rid


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_create
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegions
cdef void cregions_init(cRegions* cregions):
    cregions.id = 99999999
    cregions.n = 0
    cregions.max = 0
    cregions.regions = NULL


# :call: > --- CALLERS ---
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
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::cregion_determine_boundaries
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::cregions_get_unique_id
# :call: v stormtrack::core::typedefs::cregions_init
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


# :call: > --- CALLERS ---
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
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::grid_new_regions
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::cregion_cleanup
# :call: v stormtrack::core::typedefs::cregions_extend
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_link_region
# :call: v --- CALLING ---
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cregions_merge_connected_inplace
# :call: v --- CALLING ---
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::cregion_reset
cdef void cregions_reset(cRegions* cregions):
    # print("< cregions_reset")
    cdef int i_region
    for i_region in range(cregions.n):
        cregion_reset(
            cregions.regions[i_region], unlink_pixels=True, reset_connected=False,
        )
    cregions.n = 0


# :call: > --- CALLERS ---
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
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::cregion_cleanup
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: > stormtrack::core::identification::grow_cregion_rec
# :call: > stormtrack::core::typedefs::cregions_find_connected
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::_cregion_add_connected
cdef void cregions_connect(cRegion* cregion1, cRegion* cregion2):
    # print(f"< cregions_connect {cregion1.id} {cregion2.id}")
    _cregion_add_connected(cregion1, cregion2)
    _cregion_add_connected(cregion2, cregion1)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::_cregion_reset_connected
# :call: v stormtrack::core::typedefs::cregions_connect
# :call: v stormtrack::core::typedefs::dbg_check_connected
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

    dbg_check_connected(cregions, _name_+"(0)") # SR_DBG_PERMANENT

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
                with cdivision(True):
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
                    cregions_connect(cregion, neighbor.region)

    dbg_check_connected(cregions, _name_+"(1)")  # SR_DBG_PERMANENT


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_find_connected
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# SR_DBG_PERMANENT
cdef void dbg_check_connected(cRegions* cregions, str msg) except *:
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::typedefs::_cpixel_unlink_region
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: > stormtrack::core::typedefs::_cregion_reconnect_pixel
# :call: > stormtrack::core::typedefs::cregion_insert_pixel
# :call: > stormtrack::core::typedefs::cregion_insert_pixel_nogil
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
cdef void cpixel_set_region(cPixel* cpixel, cRegion* cregion) nogil:
    cdef bint debug = False
    # DBG_BLOCK <
    if debug:
        with gil:
            log.debug(
                f"cpixel_set_region: ({cpixel.x}, {cpixel.y})->"
                f"[{'NULL' if cregion is NULL else cregion.id}]"
            )
    # DBG_BLOCK >
    cpixel.region = cregion


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::grid_reset
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::pixeltype::pixeltype_none
cdef void cpixels_reset(cPixel** cpixels, np.int32_t nx, np.int32_t ny):
    cdef bint debug = False
    if debug:
        log.debug(f"cpixels_reset: {nx}x{ny}")
    cdef int x
    cdef int y
    for x in prange(nx, nogil=True):
        for y in range(ny):
            cpixels[x][y].region = NULL
            cpixels[x][y].is_feature_boundary = False
            cpixels[x][y].type = pixeltype_none


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_create_pixels
# :call: > stormtrack::core::typedefs::grid_create_pixels
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::pixeltype::pixeltype_none
cdef cPixel* cpixel2d_create(int n) nogil:
    cdef bint debug = False
    # DBG_BLOCK <
    if debug:
        with gil:
            log.debug(f"cpixel2d_create: {n}")
    # DBG_BLOCK >
    cdef int i
    cdef int j
    cdef cPixel* cpixels = <cPixel*>malloc(n*sizeof(cPixel))
    cdef cPixel* cpixel
    for i in prange(n):
        cpixel = &cpixels[i]
        cpixel.id = -1
        cpixel.x = -1
        cpixel.y = -1
        cpixel.v = -1
        cpixel.type = pixeltype_none
        cpixel.connectivity = 0
        cpixel.neighbors_max = 8
        for j in range(8):
            cpixel.neighbors[j] = NULL
        cpixel.region = NULL
        cpixel.is_seed = False
        cpixel.is_feature_boundary = False
    return cpixels


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::grid_new_region
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionsStore
# :call: v stormtrack::core::typedefs::cregions_store_extend
cdef cRegion* cregions_store_get_new_region(cRegionsStore* store):

    # Assess current situation
    cdef bint no_blocks = (store.n_blocks == 0)
    cdef bint in_last_block = (store.i_block + 1 == store.n_blocks)
    cdef bint at_end_of_block = (store.i_next_region == store.block_size)
    cdef bint blocks_full = (in_last_block and at_end_of_block)

    # Allocate new block if necessary
    if no_blocks or blocks_full:
        cregions_store_extend(store)
        no_blocks = False
        in_last_block = True
        at_end_of_block = False
        blocks_full = False

    # Pick next available region
    cdef cRegion* cregion
    if not at_end_of_block:
        cregion = &store.blocks[store.i_block][store.i_next_region]
        store.i_next_region += 1
    elif not in_last_block:
        store.i_block += 1
        cregion = &store.blocks[store.i_block][0]
        store.i_next_region = 1
    else:
        _name_ = "cregions_store_get_new_region"
        log.error(f"{_name_}: should not happen")
        # SR_DBG <
        log.debug("")
        log.debug("no_blocks {no_blocks}")
        log.debug("in_last_block {in_last_block}")
        log.debug("at_end_of_block {at_end_of_block}")
        log.debug("blocks_full {blocks_full}")
        # SR_DBG >
        exit(4)
    return cregion


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::grid_reset
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegionsStore
# :call: v stormtrack::core::typedefs::cregion_reset
cdef void cregions_store_reset(cRegionsStore* store):
    cdef int i
    cdef int j
    for i in range(store.n_blocks):
        for j in range(store.block_size):
            cregion_reset(
                &store.blocks[i][j], unlink_pixels=True, reset_connected=True,
            )
    store.i_block = 0
    store.i_next_region = 0


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_store_get_new_region
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegionsStore
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::typedefs::cregion_get_unique_id
# :call: v stormtrack::core::typedefs::cregion_init
cdef void cregions_store_extend(cRegionsStore* store):
    cdef int i
    cdef int nold = store.n_blocks
    cdef int nnew = nold + 1
    cdef cRegion** blocks_tmp
    # print("< cregions_store_extend {nold} -> {nnew}")
    if nold > 0:
        blocks_tmp = <cRegion**>malloc(nold*sizeof(cRegion*))
        for i in range(nold):
            blocks_tmp[i] = store.blocks[i]

    if store.blocks is not NULL:
        free(store.blocks)
    store.blocks = <cRegion**>malloc(nnew*sizeof(cRegion*))

    if nold > 0:
        for i in range(nold):
            store.blocks[i] = blocks_tmp[i]
        free(blocks_tmp)

    store.blocks[nold] = <cRegion*>malloc(store.block_size*sizeof(cRegion))
    cdef cRegionConf conf = cregion_conf_default()
    for i in range(store.block_size):
        cregion_init(&store.blocks[nold][i], conf, cregion_get_unique_id())

    store.i_block = nnew - 1
    store.n_blocks = nnew
    store.i_next_region = 0


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::grid_cleanup
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cRegionsStore
# :call: v stormtrack::core::typedefs::cregion_cleanup
cdef void cregions_store_cleanup(cRegionsStore* store):
    cdef int i
    if store.blocks is not NULL:
        for i in range(store.n_blocks):
            for j in range(store.block_size):
                cregion_cleanup(
                    &store.blocks[i][j], unlink_pixels=False, reset_connected=False,
                )
            free(store.blocks[i])
            store.blocks[i] = NULL
        free(store.blocks)
        store.blocks = NULL
    store.i_block = 0
    store.n_blocks = 0
    store.i_next_region = 0


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::identify_features
# :call: > stormtrack::core::tracking::FeatureTracker::__cinit__
# :call: > stormtrack::core::tracking::FeatureTracker::_swap_grids
# :call: > stormtrack::track_features::*
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: v stormtrack::core::tables::pixel_region_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_reset
# :call: v stormtrack::core::tables::pixel_status_table_alloc
# :call: v stormtrack::core::tables::pixel_status_table_reset
# :call: v stormtrack::core::typedefs::Constants
# :call: v stormtrack::core::typedefs::grid_cleanup
# :call: v stormtrack::core::typedefs::grid_create
# :call: v stormtrack::core::typedefs::grid_reset
# :call: v stormtrack::core::typedefs::grid_set_values
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::typedefs::Grid::__cinit__
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::grid_create_empty
# :call: v stormtrack::core::typedefs::grid_create_pixels
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
# :call: > stormtrack::core::typedefs::cpixels_reset
# :call: > stormtrack::core::typedefs::cregions_store_reset
# :call: v --- CALLERS ---
# :call: v stormtrack::core::typedefs::Grid::reset
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::typedefs::Grid::__dealloc__
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::tables::neighbor_link_stat_table_cleanup
# :call: v stormtrack::core::tables::pixel_done_table_cleanup
# :call: v stormtrack::core::tables::pixel_region_table_cleanup
# :call: v stormtrack::core::tables::pixel_status_table_cleanup
# :call: v stormtrack::core::typedefs::cregions_store_cleanup
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::grid_create
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::typedefs::_collect_neighbors
# :call: v stormtrack::core::typedefs::cpixel2d_create
@boundscheck(False)
@wraparound(False)
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::Grid::set_values
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
@boundscheck(False)
@wraparound(False)
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


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::grid_new_regions
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::typedefs::cregions_store_get_new_region
cdef cRegion* grid_new_region(cGrid* grid) except *:
    return cregions_store_get_new_region(&grid._regions)


# :call: > --- CALLERS ---
# :call: v --- CALLING ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegions
# :call: v stormtrack::core::typedefs::cregions_create
# :call: v stormtrack::core::typedefs::cregions_link_region
# :call: v stormtrack::core::typedefs::grid_new_region
cdef cRegions grid_new_regions(cGrid* grid, int n) except *:
    cdef int i
    cdef cRegions cregions = cregions_create(n)
    cdef cRegion* cregion
    for i in range(n):
        cregion = grid_new_region(grid)
        cregions_link_region(
            &cregions, cregion, cleanup=False, unlink_pixels=False,
        )
    return cregions
