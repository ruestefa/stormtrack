
cimport numpy as np

#==============================================================================

from .typedefs cimport dbg_check_connected

from .structs cimport(
        cConstants,
        cGrid,
        cPixel,
        cRegionConf,
        cregion_conf_default,
        cRegion,
        cRegions,
    )

from .typedefs cimport(
        Constants,
        Grid,
        pixeltype,
        pixeltype_none,
        pixeltype_background,
        pixeltype_feature,
        grid_create,
        grid_new_region,
        grid_new_regions,
        grid_cleanup,
        cpixel_set_region,
        cpixel_get_neighbor,
    )

from .typedefs cimport(
        cregions_create,
        cregions_cleanup,
        cregions_reset,
        cregions_link_region,
        cregions_connect,
        cregions_determine_boundaries,
        cregions_move,
        cregions_find_connected,
    )

from .typedefs cimport(
        cregion_get_unique_id,
        cregion_init,
        cregion_reset,
        cregion_cleanup,
        cregion_insert_pixel,
        cregion_insert_pixels_coords,
        cregion_determine_boundaries,
        cregion_merge,
        cregion_remove_pixel,
        cregion_overlap_n_mask,
    )

from .utilities cimport(
        NAN_UI64,
    )

#SR_TMP_NOGIL
from .typedefs cimport(
        cregion_insert_pixel_nogil,
    )

from .structs cimport(
        PixelRegionTable,
        PixelStatusTable,
        PixelDoneTable,
        NeighborLinkStatTable,
    )

from .tables cimport(
        pixel_region_table_alloc,
        pixel_region_table_reset,
        pixel_region_table_alloc_grid,
        pixel_region_table_alloc_pixels,
        pixel_region_table_grow,
        pixel_region_table_reset,
        pixel_region_table_reset_region,
        pixel_region_table_reset_slots,
        pixel_region_table_init_regions,
        pixel_region_table_insert_region,
    )

from .tables cimport(
        cRegionRankSlot,
        cRegionRankSlots,
        cregion_rank_slots_insert_region,
        cregion_rank_slots_reset,
        cregion_rank_slots_copy,
    )

from .tables cimport(
        pixel_status_table_init_feature,
        pixel_status_table_reset_feature,
        pixel_status_table_alloc,
        pixel_status_table_reset,
    )

from .tables cimport(
        pixel_done_table_alloc,
        pixel_done_table_init,
        pixel_done_table_reset,
    )

from .tables cimport(
        neighbor_link_stat_table_alloc,
        neighbor_link_stat_table_reset,
        neighbor_link_stat_table_reset_pixels,
    )

#==============================================================================

cdef void cregions_merge_connected_inplace(cRegions* cregions, cGrid* grid,
        bint exclude_seed_points, int nmax_cfeatures,
        cRegionConf cfeature_conf, cConstants *constants,
    ) except *
cdef Feature create_feature(cPixel** pixels_feature, int n_pixels_feature,
        np.int32_t[:] center, np.int32_t[:, :] extrema, np.uint64_t feature_id
    )
cdef void features_to_cregions(list regions_list, int n_regions,
        cRegions* cregions, cRegionConf cregion_conf,
        bint ignore_missing_neighbors, cGrid *grid, cConstants *constants,
        bint reuse_feature_cregion=?,
    ) except *
cdef list cregions_create_features(cRegions* cregions, np.uint64_t base_id,
        bint ignore_missing_neighbors, cGrid *grid, cConstants *constants,
        list used_ids=?,
    )

#==============================================================================

cdef struct cField2D:
    cPixel** pixels
    np.int32_t nx
    np.int32_t ny

#==============================================================================

cdef class Field2D:
    cdef readonly:
        np.int32_t nx, ny
        list _fld
    cdef list get_neighbors(Field2D self, Pixel pixel)
    cpdef list pixels(Field2D self)
    cdef Pixel get(Field2D self, int i, int j)

cdef class Pixel:
    cdef readonly:
        np.uint64_t id
        np.int32_t x, y
        double v
        Field2D fld
    cdef public:
        pixeltype type
        int region
    cpdef list neighbors(Pixel self)

cpdef Feature Feature_rebuild(np.ndarray values, np.ndarray pixels,
        np.ndarray center, np.ndarray extrema, list shells, list holes,
        np.uint64_t id, np.uint64_t track_id, str vertex_name,
        np.uint64_t timestep, dict properties, dict associates,
        list neighbors, dict _cache,
        dict _shared_boundary_pixels, dict _shared_boundary_pixels_unique
    )

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
        dict properties #SR_TODO rename to attributes
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
    cpdef void set_shells(self, list shells, bint derive=?, bint derive_pixels=?,
            bint derive_center=?, int nx=?, int ny=?) except *
    cpdef void set_holes(self, list holes, bint derive=?) except *
    cpdef void set_center(self, np.ndarray[np.int32_t, ndim=1] center, bint derive=?) except *
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
    cpdef np.ndarray to_field(self, int nx, int ny, np.float32_t bg=?, bint fill_holes=?, bint cache=?)
    cpdef np.ndarray to_mask(self, int nx, int ny, bint fill_holes=?, object type_=?, bint cache=?)
    cpdef object vertex(self)
    cpdef void set_vertex(self, object vertex) except *
    cpdef void reset_vertex(self) except *

cpdef void features_reset_cregion(list features, bint warn=?) except *

cpdef void features_find_neighbors(
        list features, Constants constants=?, np.int32_t nx=?, np.int32_t ny=?,
    )
