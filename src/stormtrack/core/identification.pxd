
# Third-party
cimport numpy as np

# Local
from .constants cimport Constants
from .constants cimport default_constants
from .cregion cimport cpixel_get_neighbor
from .cregion cimport cpixel_set_region
from .cregion cimport cregion_cleanup
from .cregion cimport cregion_connect_with
from .cregion cimport cregion_get_unique_id
from .cregion cimport cregion_init
from .cregion cimport cregion_insert_pixel
from .cregion cimport cregion_insert_pixel_nogil  # SR_TMP_NOGIL
from .cregion cimport cregion_insert_pixels_coords
from .cregion cimport cregion_merge
from .cregion cimport cregion_overlap_n_mask
from .cregion cimport cregion_remove_pixel
from .cregion cimport cregion_reset
from .cregion_boundaries cimport cregion_determine_boundaries
from .cregion_boundaries cimport cregions_determine_boundaries
from .cregions cimport cregions_cleanup
from .cregions cimport cregions_create
from .cregions cimport cregions_find_connected
from .cregions cimport cregions_link_region
from .cregions cimport cregions_move
from .cregions cimport cregions_reset
from .grid cimport Grid
from .grid cimport grid_cleanup
from .grid cimport grid_create
from .grid cimport grid_create_cregion
from .structs cimport cConstants
from .structs cimport cGrid
from .structs cimport cPixel
from .structs cimport cRegion
from .structs cimport cRegionConf
from .structs cimport cRegions
from .structs cimport cregion_conf_default
from .structs cimport pixeltype
from .structs cimport pixeltype_background
from .structs cimport pixeltype_feature
from .structs cimport pixeltype_none
from .tables cimport cRegionRankSlot
from .tables cimport cRegionRankSlots
from .tables cimport cregion_rank_slots_copy
from .tables cimport cregion_rank_slots_insert_region
from .tables cimport cregion_rank_slots_reset
from .tables cimport neighbor_link_stat_table_alloc
from .tables cimport neighbor_link_stat_table_reset
from .tables cimport neighbor_link_stat_table_reset_pixels
from .tables cimport pixel_done_table_alloc
from .tables cimport pixel_done_table_init
from .tables cimport pixel_done_table_reset
from .tables cimport pixel_region_table_alloc_grid
from .tables cimport pixel_region_table_alloc_pixels
from .tables cimport pixel_region_table_grow
from .tables cimport pixel_region_table_init_regions
from .tables cimport pixel_region_table_insert_region
from .tables cimport pixel_region_table_reset
from .tables cimport pixel_region_table_reset_region
from .tables cimport pixel_region_table_reset_slots
from .tables cimport pixel_status_table_alloc
from .tables cimport pixel_status_table_init_feature
from .tables cimport pixel_status_table_reset
from .tables cimport pixel_status_table_reset_feature
from .utilities cimport NAN_UI64


# :call: > --- callers ---
# :call: > stormtrack::extra::front_surgery::*
cdef void cregions_merge_connected_inplace(
    cRegions* cregions,
    cGrid* grid,
    bint exclude_seed_points,
    int nmax_cfeatures,
    cRegionConf cfeature_conf,
    cConstants *constants,
) except *


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::FeatureTracker::extend_tracks
# :call: > test_stormtrack::test_core::test_features::test_features::*
cpdef list features_grow(
    int n,
    list features,
    Constants constants,
    bint inplace=?,
    bint retain_orig=?,
    np.uint64_t base_id=?,
    list used_ids=?,
    Grid grid=?,
)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::tracking::FeatureTracker::extend_tracks
# :call: > stormtrack::extra::front_surgery::*
cdef void features_to_cregions(
    list regions_list,
    int n_regions,
    cRegions* cregions,
    cRegionConf cregion_conf,
    bint ignore_missing_neighbors,
    cGrid *grid,
    cConstants *constants,
    bint reuse_feature_cregion=?,
) except *


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::extra::front_surgery::*
cdef list cregions_create_features(
    cRegions* cregions,
    np.uint64_t base_id,
    bint ignore_missing_neighbors,
    cGrid *grid,
    cConstants *constants,
        list used_ids=?,
)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::Feature_rebuild
# :call: > stormtrack::core::identification::_replace_feature_associations
# :call: > stormtrack::core::identification::associate_features
# :call: > stormtrack::core::identification::create_feature
# :call: > stormtrack::core::identification::cregion_find_corresponding_feature
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::cregions_create_features
# :call: > stormtrack::core::identification::cyclones_to_features
# :call: > stormtrack::core::identification::feature2d_from_jdat
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::features_associates_obj2id
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::features_neighbors_id2obj
# :call: > stormtrack::core::identification::features_neighbors_obj2id
# :call: > stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: > stormtrack::core::identification::features_reset_cregion
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::resolve_indirect_associations
# :call: > stormtrack::core::io::rebuild_features_core
# :call: > stormtrack::core::tracking::FeatureTrack::features_ts_ns
# :call: > stormtrack::core::tracking::FeatureTracker::_assign_successors
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
# :call: > stormtrack::core::tracking::FeatureTracker::_finish_track
# :call: > stormtrack::core::tracking::FeatureTracker::_start_track
# :call: > stormtrack::core::tracking::FeatureTracker::_swap_grids
# :call: > stormtrack::core::tracking::FeatureTracker::extend_tracks
# :call: > stormtrack::core::tracking::TrackFeatureMerger::_collect_merge_es_attrs_core
# :call: > stormtrack::core::tracking::TrackFeatureMerger::collect_merge_es_attrs
# :call: > stormtrack::core::tracking::TrackFeatureMerger::collect_merge_vs_attrs
# :call: > stormtrack::core::tracking::TrackFeatureMerger::collect_neighbors
# :call: > stormtrack::core::tracking::TrackFeatureMerger::collect_vertices_edges
# :call: > stormtrack::core::tracking::TrackFeatureMerger::merge_feature
# :call: > stormtrack::core::tracking::TrackFeatureMerger::replace_vertices_edges
# :call: > stormtrack::core::tracking::TrackFeatureMerger::run
# :call: > stormtrack::core::tracking::TrackableFeatureCombination_Oldstyle::overlaps
# :call: > stormtrack::core::tracking::TrackableFeature_Oldstyle
# :call: > stormtrack::core::tracking::dbg_check_features_cregion_pixels
# :call: > stormtrack::extra::front_surgery::*
# :call: > test_stormtrack::test_core::test_features::test_area_lonlat::*
# :call: > test_stormtrack::test_core::test_features::test_features::*
# :call: > test_stormtrack::test_core::test_features::test_split_regiongrow::*
# :call: > test_stormtrack::test_core::test_tracking::test_split_tracks::*
# :call: > test_stormtrack::utils::*
cdef class Feature:
    cdef readonly:
        np.uint64_t id
        int n
        np.ndarray pixels
        np.ndarray center
        np.ndarray extrema
        np.ndarray values
        np.uint64_t _track_id
        str _vertex_name
        dict _cache

    cdef public:
        np.uint64_t timestep
        dict properties # SR_TODO rename to attributes
        dict associates
        list _stats_lst
        list neighbors
        list shells
        list holes
        dict _shared_boundary_pixels
        dict _shared_boundary_pixels_unique
        bint debug

    cdef:
        cRegion* cregion
        cRegion _cregion

    cdef void set_cregion(self, cRegion* cregion) except *

    cpdef void reset_cregion(self, bint warn=?) except *

    cpdef void cleanup_cregion(self, unlink_pixels=?, reset_connected=?) except *

    cpdef void set_id(self, np.uint64_t id_) except *

    cpdef void set_pixels(self, np.ndarray[np.int32_t, ndim=2] pixels) except *

    cpdef void set_values(self, np.ndarray[np.float32_t, ndim=1] values) except *

    cpdef void set_shells(
        self,
        list shells,
        bint derive=?,
        bint derive_pixels=?,
        bint derive_center=?,
        int nx=?,
        int ny=?,
    ) except *

    cpdef void set_holes(self, list holes, bint derive=?) except *

    cpdef void set_center(
        self, np.ndarray[np.int32_t, ndim=1] center, bint derive=?,
    ) except *

    cpdef void set_extrema(self, np.ndarray[np.int32_t, ndim=2] extrema) except *

    cpdef np.float32_t sum(self, bint abs=?) except -999

    cpdef np.float32_t min(self, bint abs=?) except -999

    cpdef np.float32_t mean(self, bint abs=?) except -999

    cpdef np.float32_t median(self, bint abs=?) except -999

    cpdef np.float32_t max(self, bint abs=?) except -999

    cpdef object track(self)

    cpdef void set_track(self, object track) except *

    cpdef void unset_track(self) except *

    cpdef void hardcode_n_pixels(self, int n) except *

    cpdef np.float32_t distance(self, Feature other) except -999

    cpdef bint overlaps(self, Feature other) except -1

    cpdef bint overlaps_bbox(self, Feature other) except -1

    cpdef tuple bbox(self)

    cpdef np.ndarray[np.int32_t, ndim=2] overlap_bbox(self, Feature other)

    cpdef list find_overlapping(self, list others)

    cpdef int overlap_n(self, Feature other) except -999

    cdef int _overlap_n_core(self,
            np.int32_t[:, :] pixels0,
            int n0,
            np.int32_t[:, :] pixels1,
            int n1,
            np.int32_t[:, :] bbox,
        ) except -999

    cpdef np.ndarray to_field(
        self, int nx, int ny, np.float32_t bg=?, bint fill_holes=?, bint cache=?,
    )

    cpdef np.ndarray to_mask(
        self, int nx, int ny, bint fill_holes=?, object type_=?, bint cache=?,
    )

    cpdef object vertex(self)

    cpdef void set_vertex(self, object vertex) except *

    cpdef void reset_vertex(self) except *


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::extra::front_surgery::*
cpdef void features_reset_cregion(list features, bint warn=?) except *


# :call: > --- callers ---
# :call: > stormtrack::extra::front_surgery::*
# :call: > test_stormtrack::test_core::test_features::test_features::*
cpdef void features_find_neighbors(
        list features, Constants constants=?, np.int32_t nx=?, np.int32_t ny=?,
    )
