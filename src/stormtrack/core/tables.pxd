
cimport numpy as np

#==============================================================================

from .structs cimport(
        cConstants,
        cPixel,
        cRegion,
        cRegions,
    )

from .structs cimport get_matching_neighbor_id

#==============================================================================

from .structs cimport PixelDoneTable

cdef void pixel_done_table_alloc(PixelDoneTable* table, cConstants* constants)
cdef void pixel_done_table_init(PixelDoneTable table, cRegion* cregion,
        np.float32_t level)
cdef void pixel_done_table_reset(PixelDoneTable table, cRegion* cregion)
cdef void pixel_done_table_cleanup(PixelDoneTable table, cConstants* constants)

#==============================================================================

from .structs cimport(
        cRegionRankSlots,
        cRegionRankSlot,
        PixelRegionTable,
    )

cdef void pixel_region_table_alloc(PixelRegionTable* table, int n_slots,
        cConstants* constants)
cdef void pixel_region_table_alloc_grid(
        PixelRegionTable* table,
        cConstants* constants,
    ) except *
cdef void pixel_region_table_alloc_pixels(
        PixelRegionTable table,
        int n_slots,
        cRegion* cregion,
    )
cdef void pixel_region_table_init_regions(
        PixelRegionTable table,
        cRegions* cregions_pixels,
        cRegions* cregions_target,
        int n_slots_max,
    )
cdef void pixel_region_table_insert_region(
        PixelRegionTable table,
        np.int32_t x,
        np.int32_t y,
        cRegion* cregion,
        np.int8_t rank,
    )
cdef void pixel_region_table_reset(
        PixelRegionTable table,
        np.int32_t x,
        np.int32_t y,
    )
cdef void pixel_region_table_reset_regions(
        PixelRegionTable table,
        cRegions* cregions,
    )
cdef void pixel_region_table_reset_slots(
        PixelRegionTable table,
        np.int32_t x,
        np.int32_t y,
    ) nogil
cdef void pixel_region_table_grow(
        PixelRegionTable table,
        cRegion* cfeature,
        int n_slots_new,
    )
cdef void pixel_region_table_reset_region(
        PixelRegionTable table,
        cRegion* cregion,
    )
cdef void pixel_region_table_cleanup(
        PixelRegionTable table,
        np.int32_t nx,
        np.int32_t ny,
    )
cdef void cregion_rank_slots_insert_region(
        cRegionRankSlots* slots,
        cRegion* cregion,
        np.int8_t rank,
    )
cdef void cregion_rank_slots_reset(
        cRegionRankSlots* slots,
    ) nogil
cdef cRegionRankSlots cregion_rank_slots_copy(
        cRegionRankSlots* slots,
    )

#==============================================================================

from .structs cimport PixelStatusTable

cdef void pixel_status_table_init_feature(
        PixelStatusTable table,
        cRegion* cfeature,
        cRegions* cregions_seeds,
    )
cdef void pixel_status_table_reset_feature(
        PixelStatusTable table,
        cRegion* cfeature,
    )
cdef void pixel_status_table_alloc(
        PixelStatusTable* table,
        cConstants* constants,
    )
cdef void pixel_status_table_cleanup(
        PixelStatusTable table,
        np.int32_t nx,
    )
cdef void pixel_status_table_reset(
        PixelStatusTable table,
        np.int32_t nx,
        np.int32_t ny,
    )

#==============================================================================

from .structs cimport NeighborLinkStatTable

cdef void neighbor_link_stat_table_alloc(
        NeighborLinkStatTable* table,
        cConstants* constants,
    ) except *
cdef void neighbor_link_stat_table_reset(
        NeighborLinkStatTable table,
        cConstants* constants,
    ) except *
cdef void neighbor_link_stat_table_reset_pixels(
        NeighborLinkStatTable table,
        cRegion* cregion,
        int n_neighbors_max,
    ) except *
cdef void neighbor_link_stat_table_cleanup(
        NeighborLinkStatTable table,
        np.int32_t nx,
        np.int32_t ny,
    ) except *
cdef void neighbor_link_stat_table_init(
        NeighborLinkStatTable table,
        cRegion* boundary_pixels,
        cConstants* constants,
    ) except *

#==============================================================================
