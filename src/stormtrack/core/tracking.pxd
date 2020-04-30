
# Third-party
cimport cython
cimport numpy as np

# Local
from .identification cimport Feature
from .identification cimport features_grow
from .identification cimport features_to_cregions
from .structs cimport cGrid
from .structs cimport SuccessorCandidate
from .structs cimport SuccessorCandidates
from .tables cimport PixelRegionTable
from .tables cimport pixel_region_table_init_regions
from .typedefs cimport Constants
from .typedefs cimport Grid
from .typedefs cimport cPixel
from .typedefs cimport cRegion
from .typedefs cimport cRegions
from .typedefs cimport cregion_conf_default
from .typedefs cimport cregion_overlap_n_tables
from .typedefs cimport cregion_overlaps_tables
from .typedefs cimport cregions_create
from .typedefs cimport cregions_link_region
from .typedefs cimport grid_create_empty


# :call: > --- callers ---
# :call: > test_stormtrack::test_core::test_tracking::test_branched_tracks::*
# :call: > test_stormtrack::test_core::test_tracking::test_simple_tracks::*
# :call: > test_stormtrack::test_core::test_tracking::test_split_tracks::*
cdef class FeatureTracker:
    cdef public:
        float f_overlap
        float f_size
        float threshold
        float min_p_tot
        float min_p_overlap
        float min_p_size
        int minsize
        int maxsize
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
        np.ndarray lon2d
        np.ndarray lat2d
        np.ndarray lon1d
        np.ndarray lat1d
        str         ts_fmt
        int nx
        int ny

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

    cdef  void _find_successor_candidates(
        self,
        cRegion* parent,
        cRegions* cfeatures,
        cRegions *candidates,
        int directionk,
    )

    cdef  void _find_successor_candidate_combinations(
        self,
        cRegions* parents,
        cRegions* cfeatures,
        SuccessorCandidates *combinations,
        int direction,
    ) except *

    cdef  int _combine_candidates(
        self,
        cRegion* parent,
        cRegions* candidates,
        SuccessorCandidates* combinations,
        int direction,
    ) except -1

    cdef  void _compute_successor_probabilities(
        self, SuccessorCandidates* combinations,
    ) except *

    cdef  void _sort_successor_candidates(
        self, SuccessorCandidates* combinations,
    ) except *

    cdef  list _assign_successors(
        self, list features, cRegions* cfeatures, SuccessorCandidates* combinations,
    )

    cpdef int split_active_tracks(self) except *

    cdef  object _merge_tracks(self, list tracks)


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::FeatureTrack::split
# :call: > stormtrack::core::tracking::FeatureTrack_rebuild
# :call: > test_stormtrack::test_core::test_tracking::test_split_tracks::*
cdef class FeatureTrackSplitter:
    cdef public:
        set used_ids

    cdef readonly:
        dict track_config

    cdef:
        bint debug


# :call: > --- callers ---
# :call: > stormtrack::core::io::rebuild_tracks
# :call: > stormtrack::core::tracking::FeatureTrackSplitter::_split_graph
# :call: > stormtrack::core::tracking::FeatureTrackSplitter::split
# :call: > stormtrack::core::tracking::TrackFeatureMerger::__cinit__
# :call: > stormtrack::core::tracking::remerge_partial_tracks
# :call: > test_stormtrack::test_core::test_tracking::test_merge_features::*
# :call: > test_stormtrack::test_core::test_tracking::test_split_tracks::*
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


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::FeatureTrack::merge_features
# :call: > test_stormtrack::test_core::test_tracking::test_merge_features::*
cdef class TrackFeatureMerger:
    cdef readonly:
        FeatureTrack track
        int nx
        int ny
        int connectivity
        set vs_attrs
        set es_attrs
        bint debug
        np.uint64_t timestep

    cpdef void run(self) except *

    cpdef void merge_feature(self, list features, list features_todo) except *

    cpdef void collect_neighbors(
        self,
        Feature feature,
        list features_todo,
        list features_orig,
        list external_neighbors,
    ) except *

    cpdef void collect_vertices_edges(
        self,
        list features_orig,
        list vs_orig,
        list vs_prev,
        list vs_next,
        list es_prev,
        list es_next,
    ) except *

    cpdef void collect_merge_vs_attrs(
        self, dict vx_attrs, list vs_orig, Feature merged_feature,
    ) except *

    cpdef void collect_merge_es_attrs(
        self,
        dict es_attrs_prev,
        dict es_attrs_next,
        list vs_orig,
        list vs_prev,
        list vs_next,
        list es_prev,
        list es_next,
        Feature merged_feature,
    ) except *

    cpdef void _collect_merge_es_attrs_core(
        self,
        dict es_attrs,
        list es_dir,
        list vs_dir,
        list vs_orig,
        Feature merged_feature,
        int direction,
    ) except *

    cpdef void replace_vertices_edges(
        self, Feature merged_feature,
        list features_orig,
        list vs_orig,
        list vs_prev,
        list vs_next,
        list es_prev,
        list es_next,
        dict vx_attrs,
        dict es_attrs_prev,
        dict es_attrs_next,
    ) except *

    cpdef void set_feature_types(self, list vertices) except *
