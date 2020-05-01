
# Third-party
cimport numpy as np

# Local
from .structs cimport cPixel
from .structs cimport pixeltype_none


# SR_TMP <
cdef int cpixel_angle_to_neighbor(cPixel* cpixel1, cPixel* cpixel2, bint minus=?) except -1
cdef void cpixels_reset(cPixel** cpixels, np.int32_t nx, np.int32_t ny)
cdef cPixel* cpixel2d_create(int n) nogil
# SR_TMP >
