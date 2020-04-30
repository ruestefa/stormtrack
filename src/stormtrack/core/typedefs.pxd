
# Third-party
cimport numpy as np

# Local
from .structs cimport PixelRegionTable
from .structs cimport cConstants
from .structs cimport cGrid
from .structs cimport cPixel
from .structs cimport cRegion
from .structs cimport cRegionConf
from .structs cimport cRegions
from .structs cimport cRegionsStore
from .structs cimport cregion_conf_default
from .structs cimport get_matching_neighbor_id
from .structs cimport grid_create_empty
from .structs cimport pixeltype
from .structs cimport pixeltype_background
from .structs cimport pixeltype_feature
from .structs cimport pixeltype_none
from .tables cimport neighbor_link_stat_table_alloc
from .tables cimport neighbor_link_stat_table_cleanup
from .tables cimport neighbor_link_stat_table_init
from .tables cimport neighbor_link_stat_table_reset
from .tables cimport neighbor_link_stat_table_reset
from .tables cimport pixel_done_table_cleanup
from .tables cimport pixel_region_table_alloc
from .tables cimport pixel_region_table_cleanup
from .tables cimport pixel_region_table_reset
from .tables cimport pixel_status_table_alloc
from .tables cimport pixel_status_table_cleanup
from .tables cimport pixel_status_table_reset
from .tables cimport pixel_status_table_reset


# :call: > --- CALLERS ---
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


# :call: > --- CALLERS ---
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
# :call: > stormtrack::core::typedefs::Grid::__cinit__
# :call: > stormtrack::core::typedefs::default_constants
cdef class Constants:
    cdef readonly:
        int nx
        int ny
        int connectivity
        int n_neighbors_max

    cdef:
        cConstants _cconstants

    cdef cConstants* to_c(self)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::identify_features
# :call: > stormtrack::core::tracking::FeatureTracker::__cinit__
# :call: > stormtrack::core::tracking::FeatureTracker::_swap_grids
# :call: > stormtrack::track_features::*
cdef class Grid:
    cdef readonly:
        Constants constants

    cdef:
        cGrid _cgrid

    cdef cGrid* to_c(self)

    cpdef void set_values(self, np.ndarray[np.float32_t, ndim=2] fld)

    cpdef void reset(self)

    cdef void reset_tables(self)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::typedefs::Grid::__cinit__
# :call: > stormtrack::extra::front_surgery::*
cdef cGrid grid_create(np.float32_t[:, :] fld, cConstants constants) except *


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::grid_create
# :call: > stormtrack::extra::front_surgery::*
cdef void grid_create_pixels(cGrid* grid, np.float32_t[:, :] fld) except *


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::Grid::set_values
cdef void grid_set_values(cGrid* grid, np.float32_t[:, :] fld) except *


# :call: > --- CALLING ---
# :call: > stormtrack::core::structs::cGrid
# :call: > stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: > stormtrack::core::tables::pixel_region_table_reset
# :call: > stormtrack::core::tables::pixel_status_table_reset
# :call: > stormtrack::core::typedefs::cpixels_reset
# :call: > stormtrack::core::typedefs::cregions_store_reset
# :call: > stormtrack::extra::front_surgery::*
cdef void grid_reset(cGrid* grid) except *


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::typedefs::Grid::__dealloc__
# :call: > stormtrack::extra::front_surgery::*
cdef void grid_cleanup(cGrid* grid) except *


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::grid_new_regions
cdef cRegion* grid_new_region(cGrid* grid) except *


# :call: > --- CALLERS ---
cdef cRegions grid_new_regions(cGrid* grid, int n) except *


cdef int CREGION_NEXT_ID = 0


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::typedefs::_determine_boundary_pixels_raw
# :call: > stormtrack::core::typedefs::cregions_store_extend
cdef np.uint64_t cregion_get_unique_id()


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::typedefs::_determine_boundary_pixels_raw
# :call: > stormtrack::core::typedefs::cregions_store_extend
cdef void cregion_init(cRegion* cregion, cRegionConf cregion_conf, np.uint64_t rid)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::typedefs::cregion_cleanup
# :call: > stormtrack::core::typedefs::cregions_reset
# :call: > stormtrack::core::typedefs::cregions_store_reset
cdef void cregion_reset(cRegion* cregion, bint unlink_pixels, bint reset_connected)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::Feature::cleanup_cregion
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::eliminate_regions_by_size
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::cregion_merge
# :call: > stormtrack::core::typedefs::cregions_cleanup
# :call: > stormtrack::core::typedefs::cregions_link_region
# :call: > stormtrack::core::typedefs::cregions_store_cleanup
cdef void cregion_cleanup(cRegion* cregion, bint unlink_pixels, bint reset_connected)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::pixels_find_boundaries
cdef void cregion_insert_pixels_coords(
    cRegion* cregion,
    cPixel** cpixels,
    np.ndarray[np.int32_t, ndim=2] coords,
    bint link_region,
    bint unlink_pixels,
) except *


# :call: > --- CALLERS ---
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
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::cregion_insert_pixels_coords
# :call: > stormtrack::core::typedefs::cregion_merge
cdef void cregion_insert_pixel(
    cRegion* cregion, cPixel* cpixel, bint link_region, bint unlink_pixel,
)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_find_background_neighbor_pixels
# SR_TMP_NOGIL <<<
cdef void cregion_insert_pixel_nogil(
    cRegion* cregion, cPixel* cpixel, bint link_region, bint unlink_pixel,
) nogil


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::typedefs::cregion_insert_pixel
cdef void cregion_remove_pixel(cRegion* cregion, cPixel* cpixel)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_reset_connected
cdef void cregion_remove_connected(cRegion* cregion, cRegion* cregion_other)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::find_existing_region
cdef cRegion* cregion_merge(cRegion* cregion1, cRegion* cregion2)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_determine_boundaries
cdef void cregion_reset_boundaries(cRegion* cregion)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::extra::front_surgery::*
cdef void cregion_determine_boundaries(cRegion* cregion, cGrid* grid) except *


# :call: > --- CALLERS ---
cdef bint cregion_overlaps(cRegion* cregion, cRegion* cregion_other)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
cdef bint cregion_overlaps_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
)


# :call: > --- CALLERS ---
# :call: > stormtrack::extra::front_surgery::*
cdef int cregion_overlap_n(cRegion* cregion, cRegion* cregion_other)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::tracking::FeatureTracker::_compute_successor_probabilities
cdef int cregion_overlap_n_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
cdef int cregion_overlap_n_mask(cRegion* cregion, np.ndarray[np.uint8_t, ndim=2] mask)


cdef int CREGIONS_NEXT_ID = 0


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_create
cdef np.uint64_t cregions_get_unique_id()


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_create
cdef void cregions_init(cRegions* cregions)


# :call: > --- CALLERS ---
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
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::cregion_determine_boundaries
# :call: > stormtrack::extra::front_surgery::*
cdef cRegions cregions_create(int nmax)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
cdef void cregions_reset(cRegions* cregions)


# :call: > --- CALLERS ---
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
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: > stormtrack::extra::front_surgery::*
cdef void cregions_cleanup(cRegions* cregions, bint cleanup_regions)


# :call: > --- CALLERS ---
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
# :call: > stormtrack::core::typedefs::_reconstruct_boundaries
# :call: > stormtrack::core::typedefs::grid_new_regions
# :call: > stormtrack::extra::front_surgery::*
cdef void cregions_link_region(
    cRegions* cregions, cRegion* cregion, bint cleanup, bint unlink_pixels,
)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cregions_merge_connected_inplace
cdef void cregions_move(cRegions* source, cRegions* target)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: > stormtrack::core::identification::grow_cregion_rec
# :call: > stormtrack::core::typedefs::cregions_find_connected
cdef void cregions_connect(cRegion* cregion1, cRegion* cregion2)


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::extra::front_surgery::*
cdef void cregions_find_connected(
    cRegions* cregions, bint reset_existing, cConstants* constants,
) except *


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::typedefs::cregion_determine_boundaries
cdef void cregions_determine_boundaries(cRegions* cregions, cGrid* grid) except *


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::cregions_find_connected
cdef void dbg_check_connected(cRegions* cregions, str msg) except *


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::typedefs::_cpixel_unlink_region
# :call: > stormtrack::core::typedefs::_cregion_determine_boundaries_core
# :call: > stormtrack::core::typedefs::_cregion_reconnect_pixel
# :call: > stormtrack::core::typedefs::cregion_insert_pixel
# :call: > stormtrack::core::typedefs::cregion_insert_pixel_nogil
cdef void cpixel_set_region(cPixel* cpixel, cRegion* cregion) nogil


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::_cregion_create_pixels
# :call: > stormtrack::core::typedefs::grid_create_pixels
cdef cPixel* cpixel2d_create(int n) nogil


# :call: > --- CALLERS ---
# :call: > stormtrack::core::typedefs::grid_reset
cdef void cpixels_reset(cPixel** cpixels, np.int32_t nx, np.int32_t ny)  # SR_TMP


# :call: > --- CALLERS ---
# :call: > stormtrack::core::identification::cpixel_count_neighbors_in_cregion
cdef cPixel* cpixel_get_neighbor(
        cPixel* cpixel,
        int index,
        cPixel** cpixels,
        np.int32_t nx,
        np.int32_t ny,
        int connectivity,
)
