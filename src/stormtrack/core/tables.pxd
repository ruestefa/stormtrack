
# Third-party
cimport numpy as np

# Local
from .structs cimport NeighborLinkStatTable
from .structs cimport PixelDoneTable
from .structs cimport PixelRegionTable
from .structs cimport PixelStatusTable
from .structs cimport cConstants
from .structs cimport cPixel
from .structs cimport cRegion
from .structs cimport cRegionRankSlot
from .structs cimport cRegionRankSlots
from .structs cimport cRegions
from .structs cimport get_matching_neighbor_id


# :call: > --- CALLERS ---
# :call: > identification::csplit_regiongrow_levels
cdef void pixel_done_table_alloc(PixelDoneTable* table, cConstants* constants)


# :call: > --- CALLERS ---
# :call: > identification::extract_subregions_level
cdef void pixel_done_table_init(
    PixelDoneTable table, cRegion* cregion, np.float32_t level,
)


# :call: > --- CALLERS ---
# :call: > identification::extract_subregions_level
cdef void pixel_done_table_reset(PixelDoneTable table, cRegion* cregion)


# :call: > --- CALLERS ---
# :call: > typedefs::grid_cleanup
cdef void pixel_done_table_cleanup(PixelDoneTable table, cConstants* constants)


# :call: > --- CALLERS ---
# :call: > typedefs::Grid::__cinit__
cdef void pixel_region_table_alloc(
    PixelRegionTable* table, int n_slots, cConstants* constants,
)


# :call: > --- CALLERS ---
# :call: > identification::_find_features_threshold_random_seeds
# :call: > identification::c_find_features_2d_threshold_seeds
# :call: > identification::csplit_regiongrow_levels
# :call: > identification::feature_split_regiongrow
# :call: > identification::features_find_neighbors
# :call: > identification::find_features_2d_threshold
# :call: > identification::merge_adjacent_features
cdef void pixel_region_table_alloc_grid(
    PixelRegionTable* table, cConstants* constants,
) except *


# :call: > --- CALLERS ---
# :call: > identification::cfeatures_grow_core
# :call: > identification::csplit_regiongrow_core
# :call: > identification::csplit_regiongrow_levels
# :call: > tables::pixel_region_table_grow
cdef void pixel_region_table_alloc_pixels(
    PixelRegionTable table, int n_slots, cRegion* cregion,
)


# :call: > --- CALLERS ---
# :call: > identification::cfeatures_grow_core
# :call: > identification::csplit_regiongrow_core
# :call: > tracking::FeatureTracker::extend_tracks
cdef void pixel_region_table_init_regions(
        PixelRegionTable table,
        cRegions* cregions_pixels,
        cRegions* cregions_target,
        int n_slots_max,
)


# :call: > --- CALLERS ---
# :call: > identification::determine_shared_boundary_pixels
# :call: > identification::regiongrow_resolve_multi_assignments
cdef void pixel_region_table_insert_region(
        PixelRegionTable table,
        np.int32_t x,
        np.int32_t y,
        cRegion* cregion,
        np.int8_t rank,
)


# :call: > --- CALLERS ---
# :call: > identification::cregions2features_connected2neighbors
# :call: > identification::find_features_2d_threshold
# :call: > typedefs::Grid::reset_tables
# :call: > typedefs::grid_reset
cdef void pixel_region_table_reset(
    PixelRegionTable table, np.int32_t x, np.int32_t y,
)


# :call: > --- CALLERS ---
cdef void pixel_region_table_reset_regions(
    PixelRegionTable table, cRegions* cregions,
)


# :call: > --- CALLERS ---
# :call: > identification::regiongrow_resolve_multi_assignments
# :call: > tables::pixel_region_table_reset
# :call: > tables::pixel_region_table_reset_region
cdef void pixel_region_table_reset_slots(
    PixelRegionTable table, np.int32_t x, np.int32_t y,
) nogil


# :call: > --- CALLERS ---
# :call: > identification::csplit_regiongrow_levels_core
cdef void pixel_region_table_grow(
    PixelRegionTable table, cRegion* cfeature, int n_slots_new,
)


# :call: > --- CALLERS ---
# :call: > identification::cfeatures_grow_core
# :call: > identification::csplit_regiongrow_core
# :call: > identification::csplit_regiongrow_levels
# :call: > tables::pixel_region_table_reset_regions
cdef void pixel_region_table_reset_region(
    PixelRegionTable table, cRegion* cregion,
)


# :call: > --- CALLERS ---
# :call: > typedefs::grid_cleanup
cdef void pixel_region_table_cleanup(
    PixelRegionTable table, np.int32_t nx, np.int32_t ny,
)


# :call: > --- CALLERS ---
# :call: > identification::resolve_multi_assignment_best_connected_region
# :call: > identification::resolve_multi_assignment_biggest_region
# :call: > identification::resolve_multi_assignment_mean_strongest_region
# :call: > identification::resolve_multi_assignment_strongest_region
# :call: > tables::pixel_region_table_insert_region
cdef void cregion_rank_slots_insert_region(
    cRegionRankSlots* slots, cRegion* cregion, np.int8_t rank,
)


# :call: > --- CALLERS ---
# :call: > identification::resolve_multi_assignment_best_connected_region
# :call: > identification::resolve_multi_assignment_biggest_region
# :call: > identification::resolve_multi_assignment_mean_strongest_region
# :call: > identification::resolve_multi_assignment_strongest_region
# :call: > tables::pixel_region_table_reset_slots
cdef void cregion_rank_slots_reset(cRegionRankSlots* slots) nogil


# :call: > --- CALLERS ---
# :call: > identification::resolve_multi_assignment_biggest_region
# :call: > identification::resolve_multi_assignment_mean_strongest_region
# :call: > identification::resolve_multi_assignment_strongest_region
cdef cRegionRankSlots cregion_rank_slots_copy(cRegionRankSlots* slots)


# :call: > --- CALLERS ---
# :call: > identification::cfeatures_grow_core
# :call: > identification::csplit_regiongrow_core
cdef void pixel_status_table_init_feature(
    PixelStatusTable table, cRegion* cfeature, cRegions* cregions_seeds,
)


# :call: > --- CALLERS ---
# :call: > identification::cfeatures_grow_core
# :call: > identification::csplit_regiongrow_core
cdef void pixel_status_table_reset_feature(PixelStatusTable table, cRegion* cfeature)


# :call: > --- CALLERS ---
# :call: > identification::_find_features_threshold_random_seeds
# :call: > identification::c_find_features_2d_threshold_seeds
# :call: > identification::csplit_regiongrow_levels
# :call: > identification::feature_split_regiongrow
# :call: > identification::find_features_2d_threshold
# :call: > typedefs::Grid::__cinit__
cdef void pixel_status_table_alloc(PixelStatusTable* table, cConstants* constants)


# :call: > --- CALLERS ---
# :call: > typedefs::grid_cleanup
cdef void pixel_status_table_cleanup(PixelStatusTable table, np.int32_t nx)


# :call: > --- CALLERS ---
# :call: > typedefs::Grid::reset_tables
# :call: > typedefs::grid_reset
cdef void pixel_status_table_reset(PixelStatusTable table, np.int32_t nx, np.int32_t ny)


# :call: > --- CALLERS ---
# :call: > identification::_find_features_threshold_random_seeds
# :call: > identification::c_find_features_2d_threshold_seeds
# :call: > identification::csplit_regiongrow_levels
# :call: > identification::feature_split_regiongrow
# :call: > identification::features_find_neighbors
# :call: > identification::find_features_2d_threshold
# :call: > identification::merge_adjacent_features
# :call: > identification::pixels_find_boundaries
# :call: > typedefs::Grid::__cinit__
cdef void neighbor_link_stat_table_alloc(
    NeighborLinkStatTable* table, cConstants* constants,
) except *


# :call: > --- CALLERS ---
# :call: > identification::find_features_2d_threshold
# :call: > typedefs::Grid::reset_tables
# :call: > typedefs::_reconstruct_boundaries
# :call: > typedefs::grid_reset
cdef void neighbor_link_stat_table_reset(
        NeighborLinkStatTable table, cConstants* constants,
) except *


# :call: > --- CALLERS ---
# :call: > identification::csplit_regiongrow_levels_core
cdef void neighbor_link_stat_table_reset_pixels(
    NeighborLinkStatTable table, cRegion* cregion, int n_neighbors_max,
) except *


# :call: > --- CALLERS ---
# :call: > typedefs::grid_cleanup
cdef void neighbor_link_stat_table_cleanup(
    NeighborLinkStatTable table, np.int32_t nx, np.int32_t ny,
) except *


# :call: > --- CALLERS ---
# :call: > typedefs::_reconstruct_boundaries
cdef void neighbor_link_stat_table_init(
    NeighborLinkStatTable table, cRegion* boundary_pixels, cConstants* constants,
) except *
