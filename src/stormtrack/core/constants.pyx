# !/usr/bin/env python3

from __future__ import print_function

# # C: C libraries
# from libc.math cimport pow
# from libc.math cimport sqrt
# from libc.stdlib cimport exit
# from libc.stdlib cimport free
# from libc.stdlib cimport malloc

# C: Third-party
cimport cython
cimport numpy as np
# from cython.parallel cimport prange

# Standard library
# import logging as log

# Third-party
import numpy as np


# :call: > --- callers ---
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
# :call: > stormtrack::extra::front_surgery::*
# :call: > test_stormtrack::test_core::test_features::test_boundaries::*
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: > test_stormtrack::test_core::test_features::test_regions::*
# :call: > test_stormtrack::test_core::test_features::test_split_regiongrow::*
# :call: > test_stormtrack::utils::*
# :call: v --- calling ---
# :call: v stormtrack::core::constants::Constants
cpdef Constants default_constants(
    # SR_TODO remove nx, ny (use from Grid)
    int nx, int ny, int connectivity=4, int n_neighbors_max=8,
):
    return Constants(
        nx=nx, ny=ny, connectivity=connectivity, n_neighbors_max=n_neighbors_max,
    )


# :call: > --- callers ---
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
# :call: > stormtrack::core::grid::Grid::__cinit__
# :call: > stormtrack::core::constants::default_constants
# :call: v --- calling ---
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
