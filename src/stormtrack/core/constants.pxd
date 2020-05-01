
# Third-party
# cimport numpy as np

# Local
from .structs cimport cConstants


# :call: > --- callers ---
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
# :call: > stormtrack::extra::front_surgery::*
# :call: > stormtrack::identify_features::*
# :call: > stormtrack::track_features::*
# :call: > test_stormtrack::test_core::test_features::test_boundaries::*
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: > test_stormtrack::test_core::test_features::test_regions::*
# :call: > test_stormtrack::test_core::test_features::test_split_regiongrow::*
# :call: > test_stormtrack::utils::*
cpdef Constants default_constants(
    int nx, int ny, int connectivity=?, int n_neighbors_max=?,
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
cdef class Constants:
    cdef readonly:
        int nx
        int ny
        int connectivity
        int n_neighbors_max

    cdef:
        cConstants _cconstants

    cdef cConstants* to_c(self)
