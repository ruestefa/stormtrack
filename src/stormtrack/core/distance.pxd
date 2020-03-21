
# Third-party
cimport numpy as np


cdef np.ndarray[np.float32_t, ndim=4] _pairwise_distances_great_circle__core(
    np.ndarray[np.float32_t, ndim=2] lon, np.ndarray[np.float32_t, ndim=2] lat,
)


cdef np.float32_t great_circle_distance(
    np.float32_t lon0, np.float32_t lat0, np.float32_t lon1, np.float32_t lat1,
)
