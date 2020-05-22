# -*- coding: utf-8 -*-
"""
TODO
"""

from __future__ import print_function

# C: C libraries
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
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
cdef int cpixel_angle_to_neighbor(
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


# :call: > --- callers ---
# :call: > stormtrack::core::grid::grid_reset
# :call: v --- calling ---
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


# :call: > --- callers ---
# :call: > stormtrack::core::grid::grid_create_pixels
# :call: v --- calling ---
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
