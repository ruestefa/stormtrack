
cimport numpy as np

#==============================================================================

from .structs cimport(
        cConstants,
        cGrid,
    )

from .typedefs cimport(
        Constants,
        Grid,
        cPixel,
        grid_create_empty,
        grid_create,
        grid_reset,
        grid_cleanup,
        grid_new_region,
        grid_new_regions,
    )

from .typedefs cimport(
        cRegion,
        cregion_cleanup,
        cregion_conf_default,
        cregion_overlaps,
        cregion_overlaps_tables,
        cregion_overlap_n,
        cregion_overlap_n_tables,
    )

from .typedefs cimport(
        cRegions,
        cregions_init,
        cregions_create,
        cregions_cleanup,
        cregions_link_region,
    )

from .tables cimport(
        neighbor_link_stat_table_alloc,
        neighbor_link_stat_table_reset,
    )

from .tables cimport(
        PixelRegionTable,
        pixel_region_table_alloc,
        pixel_region_table_init_regions,
        pixel_region_table_reset,
        pixel_region_table_reset_regions,
        pixel_region_table_cleanup,
    )

from .identification cimport(
        Feature,
        features_to_cregions,
    )

#==============================================================================

cdef struct SuccessorCandidate:
    cRegion*      parent
    cRegion**      children
    np.float32_t *p_shares
    np.int32_t   *n_overlaps
    int          n, max, direction
    float        p_tot, p_size, p_overlap

cdef struct SuccessorCandidates:
    SuccessorCandidate* candidates
    int n, max

cdef class FeatureTracker:
    cdef public:
        float f_overlap, f_size, threshold
        float min_p_tot, min_p_overlap, min_p_size
        int minsize, maxsize
        int max_children
        int split_tracks_n
        int grow_features_n
        bint merge_features
        bint reshrink_tracked_features
        set used_ids
        bint debug
    cdef readonly:
        Grid        _grid_now
        Grid        _grid_new
        Grid        _grid_grow
        np.uint64_t _ts_nan
        np.uint64_t previous_timestep
        np.uint64_t current_timestep
        list        active_tracks
        list        finished_tracks
        Constants   constants
        bint        size_lonlat
        str         area_lonlat_method
        np.ndarray  lon2d, lat2d, lon1d, lat1d
        str         ts_fmt
        int         nx, ny
    cdef:
        dict _feature_dict_new
    cpdef void extend_tracks(self, list features, np.uint64_t timestep) except *
    cpdef void finish_tracks(self) except *
    cdef  void _finish_branch(self, object vertex)
    cdef  void _finish_track(self, FeatureTrack track, bint force) except *
    cpdef list pop_finished_tracks(self)
    cpdef void reset(self)
    cdef  void _swap_grids(self) except *
    cdef  void _extend_tracks_core(self, list features, cRegions* cfeatures) except *
    cdef  FeatureTrack _start_track(self, cRegion* cfeature)
    cpdef object head_vertices(self)
    cpdef list head_features(self)
    cdef  void _find_successor_candidates(self, cRegion* parent,
            cRegions* cfeatures, cRegions *candidates, int direction)
    cdef  void _find_successor_candidate_combinations(self, cRegions* parents,
            cRegions* cfeatures, SuccessorCandidates *combinations,
            int direction) except *
    cdef  int _combine_candidates(self, cRegion* parent, cRegions* candidates,
            SuccessorCandidates* combinations, int direction) except -1
    cdef  void _compute_successor_probabilities(self,
            SuccessorCandidates* combinations) except *
    cdef  void _sort_successor_candidates(self,
            SuccessorCandidates* combinations) except *
    cdef  list _assign_successors(self, list features, cRegions* cfeatures,
            SuccessorCandidates* combinations)
    cpdef int split_active_tracks(self) except *
    cdef  object _merge_tracks(self, list tracks)

cdef void compute_tracking_probabilities(float* p_tot, float* p_size,
        float* p_overlap, int n_parent, int n_child, int n_overlap,
        float f_size, float f_overlap, float min_p_size, float min_p_overlap
    ) except *

cdef class FeatureTrackSplitter:
    cdef public:
        set used_ids
    cdef readonly:
        dict track_config
    cdef:
        bint debug

#==============================================================================

cpdef FeatureTrack FeatureTrack_rebuild(np.uint64_t id_, object graph,
        dict config)

# Info: This class has initially been written in pure Python to convert
# old-style tracks into new tracks using igraph, i.e. the class has not
# initially been designed as an extension type.
cdef class FeatureTrack:
    cdef public:
        object graph
        FeatureTracker tracker
    cdef readonly:
        np.uint64_t id
        dict config
        bint debug
        list _total_track_stats_lst
        list _missing_features_stats_lst
        list _total_track_stats_keys_fcts_lst
        list _missing_features_stats_keys_fcts_lst
        dict _cache
    cpdef list finish(self, bint force=?, int split_n=?, bint merge_features=?)
    cpdef list split(self, int n=?, used_ids=?, bint preserve=?)
    cpdef void merge_features(self, Constants constants) except *

cdef class TrackFeatureMerger:
    cdef readonly:
        FeatureTrack track
        int nx, ny, connectivity
        set vs_attrs
        set es_attrs
        bint debug
        np.uint64_t timestep
    cpdef void run(self) except *
    cpdef void merge_feature(self, list features, list features_todo) except *
    cpdef void collect_neighbors(self, Feature feature, list features_todo,
            list features_orig, list external_neighbors) except *
    cpdef void collect_vertices_edges(self, list features_orig, list vs_orig,
            list vs_prev, list vs_next, list es_prev, list es_next) except *
    cpdef void collect_merge_vs_attrs(self, dict vx_attrs, list vs_orig,
            Feature merged_feature) except *
    cpdef void collect_merge_es_attrs(self, dict es_attrs_prev,
            dict es_attrs_next, list vs_orig, list vs_prev, list vs_next,
            list es_prev, list es_next, Feature merged_feature) except *
    cpdef void _collect_merge_es_attrs_core(self, dict es_attrs, list es_dir,
            list vs_dir, list vs_orig, Feature merged_feature,
            int direction) except *
    cpdef void replace_vertices_edges(self, Feature merged_feature,
            list features_orig, list vs_orig, list vs_prev, list vs_next,
            list es_prev, list es_next, dict vx_attrs, dict es_attrs_prev,
            dict es_attrs_next) except *
    cpdef void set_feature_types(self, list vertices) except *

#==============================================================================
