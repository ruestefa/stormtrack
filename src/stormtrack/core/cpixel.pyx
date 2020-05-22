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


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
cdef bint cpixel_equals(cPixel *this, cPixel *other) except -1:
    if this.x == other.x and this.y == other.y:
        return True
    else:
        return False
