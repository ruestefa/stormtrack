
# Third-party
cimport numpy as np

# Local
from .structs cimport cConstants
from .structs cimport cGrid
from .structs cimport cPixel
from .structs cimport cRegion
from .structs cimport cRegions


# SR_TMP <
from .cregion cimport cregion_northernmost_pixel
from .cregion cimport cregion_check_validity
from .cregion cimport cregion_cleanup
from .cregion cimport _cregion_reset_connected
from .cregion cimport cregion_connect_with
from .cregion cimport cregion_reset
from .cpixel cimport cpixel_angle_to_neighbor
# SR_TMP >


cdef int CREGIONS_NEXT_ID = 0


# SR_TMP <
cdef np.uint64_t cregions_get_unique_id()
cdef void cregions_init(cRegions* cregions)
cdef bint* categorize_boundaries(cRegions* boundaries, cGrid* grid) except *
cdef void cregions_extend(cRegions* cregions)
# SR_TMP >


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::extra::front_surgery::*
cdef void cregions_find_connected(
    cRegions* cregions, bint reset_existing, cConstants* constants,
) except *


# :call: > --- callers ---
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
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::extra::front_surgery::*
cdef void cregions_cleanup(cRegions* cregions, bint cleanup_regions)


# :call: > --- callers ---
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
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: > stormtrack::extra::front_surgery::*
cdef cRegions cregions_create(int nmax)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cregions_merge_connected_inplace
cdef void cregions_move(cRegions* source, cRegions* target)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
cdef void cregions_reset(cRegions* cregions)


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
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
# :call: > stormtrack::core::tracking::FeatureTracker::_find_successor_candidates
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::extra::front_surgery::*
cdef void cregions_link_region(
    cRegions* cregions, cRegion* cregion, bint cleanup, bint unlink_pixels,
)
