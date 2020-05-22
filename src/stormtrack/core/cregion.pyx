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
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::cregions_store::cregions_store_extend
# :call: v --- calling ---
cdef np.uint64_t cregion_get_unique_id():
    global CREGION_NEXT_ID
    cdef np.uint64_t rid = CREGION_NEXT_ID
    CREGION_NEXT_ID += 1
    return rid


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::cregions_store::cregions_store_extend
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_extend_pixels
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_extend_pixels_nogil
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_shells
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_shells_nogil
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_holes
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_holes_nogil
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_remove_pixel
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
@cython.profile(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_remove_pixel_nogil
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
@cython.profile(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_remove_pixel
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_shell_remove_gaps
@cython.profile(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_remove_pixel_nogil
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_shell_remove_gaps_nogil
@cython.profile(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_remove_pixel
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_hole_remove_gaps
@cython.profile(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_remove_pixel_nogil
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_hole_remove_gaps_nogil
@cython.profile(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_insert_pixel
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_pixels_remove_gaps
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_insert_pixel_nogil
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_pixels_remove_gaps_nogil
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_insert_pixel_nogil
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_insert_hole_pixel
# :call: > stormtrack::core::cregion::_cregion_new_hole
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_new_shell
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_new_hole
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_extend_shell
# :call: v stormtrack::core::cregion::_cregion_extend_shells
cdef void _cregion_new_shell(cRegion* cregion):
    # print(f"< _cregion_new_shell {cregion.shells_n}/{cregion.shells_max}")
    if cregion.shells_max == 0:
        _cregion_extend_shells(cregion)
    if cregion.shell_max[cregion.shells_n] == 0:
        _cregion_extend_shell(cregion, cregion.shells_n)
    cregion.shells_n += 1
    if cregion.shells_n == cregion.shells_max:
        _cregion_extend_shells(cregion)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_extend_hole
# :call: v stormtrack::core::cregion::_cregion_extend_holes
cdef void _cregion_new_hole(cRegion* cregion):
    # print(f"< _cregion_new_hole {cregion.holes_n}/{cregion.holes_max}")
    if cregion.holes_max == 0:
        _cregion_extend_holes(cregion)
    if cregion.hole_max[cregion.holes_n] == 0:
        _cregion_extend_hole(cregion, cregion.holes_n)
    cregion.holes_n += 1
    if cregion.holes_n == cregion.holes_max:
        _cregion_extend_holes(cregion)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_connect_with
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_extend_connected
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_add_connected
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_reset_connected
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion_boundaries::cregions_find_northernmost_uncategorized_region
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_reset
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_remove_connected
cdef inline void _cregion_reset_connected(cRegion* cregion, bint unlink):
    cdef int i
    for i in range(cregion.connected_n):
        if unlink:
            other_cregion = cregion.connected[i]
            cregion_remove_connected(other_cregion, cregion)
        cregion.connected[i] = NULL
    cregion.connected_n = 0


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cpixel_count_neighbors_in_cregion
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::cregion::_cpixel_get_neighbor
cdef cPixel* cpixel_get_neighbor(
    cPixel* cpixel,
    int index,
    cPixel** cpixels,
    np.int32_t nx,
    np.int32_t ny,
    int connectivity,
):
    return _cpixel_get_neighbor(cpixel, index, cpixels, nx, ny, connectivity)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_collect_neighbors
# :call: > stormtrack::core::cregion::cpixel_get_neighbor
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::grid::grid_create_pixels
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::cregion::_cpixel_get_neighbor
@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
@cython.boundscheck(False)
@cython.wraparound(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_overlap_core
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::cregion::_cpixel_unlink_region
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion::_cregion_reconnect_pixel
# :call: > stormtrack::core::cregion::cregion_insert_pixel
# :call: > stormtrack::core::cregion::cregion_insert_pixel_nogil
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_insert_pixel_nogil
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_remove_pixel_from_holes_nogil
# :call: v stormtrack::core::cregion::_cregion_remove_pixel_from_pixels_nogil
# :call: v stormtrack::core::cregion::_cregion_remove_pixel_from_shells_nogil
@cython.profile(False)
cdef void cregion_remove_pixel_nogil(cRegion* cregion, cPixel* cpixel) nogil:
    _cregion_remove_pixel_from_pixels_nogil(cregion, cpixel)
    _cregion_remove_pixel_from_shells_nogil(cregion, cpixel)
    _cregion_remove_pixel_from_holes_nogil(cregion, cpixel)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cregion_insert_hole_pixel
# :call: > stormtrack::core::cregion::_cregion_insert_shell_pixel
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cpixel_set_region
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion::_cregion_extend_shell
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_extend_shell
# :call: v stormtrack::core::cregion::_cregion_reconnect_pixel
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion::_cregion_extend_hole
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_extend_hole
# :call: v stormtrack::core::cregion::_cregion_reconnect_pixel
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::cregion::cregion_insert_pixel
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_remove_pixel_from_holes
# :call: v stormtrack::core::cregion::_cregion_remove_pixel_from_pixels
# :call: v stormtrack::core::cregion::_cregion_remove_pixel_from_shells
@cython.profile(False)
cdef void cregion_remove_pixel(cRegion* cregion, cPixel* cpixel):
    cdef bint debug = False
    if debug:
        log.debug(f"cregion_remove_pixel: [{cregion.id}]</-({cpixel.x}, {cpixel.y})")
    _cregion_remove_pixel_from_pixels(cregion, cpixel)
    _cregion_remove_pixel_from_shells(cregion, cpixel)
    _cregion_remove_pixel_from_holes(cregion, cpixel)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_reset
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cpixel_set_region
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::cregion::cregion_cleanup
# :call: > stormtrack::core::cregions::cregions_reset
# :call: > stormtrack::core::cregions_store::cregions_store_reset
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cpixel_unlink_region
# :call: v stormtrack::core::cregion::_cregion_reset_connected
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::Feature::cleanup_cregion
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::eliminate_regions_by_size
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion::cregion_merge
# :call: > stormtrack::core::cregions::cregions_cleanup
# :call: > stormtrack::core::cregions::cregions_link_region
# :call: > stormtrack::core::cregions_store::cregions_store_cleanup
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_reset
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


# :call: > --- callers ---
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
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion::cregion_insert_pixels_coords
# :call: > stormtrack::core::cregion::cregion_merge
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_extend_pixels
# :call: v stormtrack::core::cregion::cpixel_set_region
# :call: v stormtrack::core::cregion::cregion_remove_pixel
@cython.profile(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_background_neighbor_pixels
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_extend_pixels_nogil
# :call: v stormtrack::core::cregion::cpixel_set_region
# :call: v stormtrack::core::cregion::cregion_remove_pixel_nogil
@cython.profile(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_insert_pixel
@cython.boundscheck(False)
@cython.wraparound(False)
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::find_existing_region
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_cleanup
# :call: v stormtrack::core::cregion::cregion_insert_pixel
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: > stormtrack::core::identification::grow_cregion_rec
# :call: > stormtrack::core::cregions::cregions_find_connected
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_add_connected
cdef void cregion_connect_with(cRegion* cregion1, cRegion* cregion2):
    # print(f"< cregion_connect_with {cregion1.id} {cregion2.id}")
    _cregion_add_connected(cregion1, cregion2)
    _cregion_add_connected(cregion2, cregion1)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::cregion::_collect_neighbors
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
# :call: v stormtrack::core::cregion::cregion_insert_pixel
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


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
# :call: v --- calling ---
# :call: v stormtrack::core::structs::PixelRegionTable
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_overlap_core
cdef bint cregion_overlaps_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
):
    return _cregion_overlap_core(
        cregion, cregion_other, table, table_other, count=False,
    )


# :call: > --- callers ---
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_overlap_core
cdef int cregion_overlap_n(cRegion* cregion, cRegion* cregion_other):
    """Count overlapping pixels between two cregions."""
    return _cregion_overlap_core(
        cregion, cregion_other, table=NULL, table_other=NULL, count=True,
    )


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::FeatureTracker::_compute_successor_probabilities
# :call: v --- calling ---
# :call: v stormtrack::core::structs::PixelRegionTable
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::_cregion_overlap_core
cdef int cregion_overlap_n_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
):
    return _cregion_overlap_core(cregion, cregion_other, table, table_other, count=True)


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::cregion_overlap_n
# :call: > stormtrack::core::cregion::cregion_overlap_n_tables
# :call: > stormtrack::core::cregion::cregion_overlaps_tables
# :call: v --- calling ---
# :call: v stormtrack::core::structs::PixelRegionTable
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::cregion::cregion_determine_bbox
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
