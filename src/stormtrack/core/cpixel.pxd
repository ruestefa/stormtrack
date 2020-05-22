# -*- coding: utf-8 -*-

# Third-party
cimport numpy as np

# Local
from .structs cimport cPixel
from .structs cimport pixeltype_none


# SR_TMP <
cdef void cpixels_reset(cPixel** cpixels, np.int32_t nx, np.int32_t ny)
cdef cPixel* cpixel2d_create(int n) nogil
cdef bint cpixel_equals(cPixel *this, cPixel *other) except -1
# SR_TMP >
