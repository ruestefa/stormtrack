
cimport numpy as np

#==============================================================================

from .structs cimport(
        cConstants,

        cPixel,

        cRegionConf,
        cregion_conf_default,

        cRegion,
        cRegions,

        cRegionsStore,
        cregions_store_create,

        cGrid,
        grid_create_empty,

        PixelRegionTable,
        PixelStatusTable,
        PixelDoneTable,
        NeighborLinkStatTable,
    )

from .structs cimport(
        pixeltype,
        pixeltype_none,
        pixeltype_background,
        pixeltype_feature,
    )

from .structs cimport get_matching_neighbor_id

from .tables cimport(
        pixel_region_table_alloc,
        pixel_region_table_reset,
        pixel_status_table_alloc,
        pixel_status_table_reset,
        neighbor_link_stat_table_alloc,
        neighbor_link_stat_table_init,
        neighbor_link_stat_table_reset,
        neighbor_link_stat_table_reset_pixels,
    )

#==============================================================================
# Constants
#==============================================================================

cdef class Constants:
    cdef readonly:
        int nx, ny, connectivity, n_neighbors_max
    cdef:
        cConstants _cconstants
    cdef cConstants* to_c(self)

#==============================================================================
# GRID
#==============================================================================

from .tables cimport(
        pixel_region_table_reset,
        pixel_status_table_reset,
        pixel_done_table_reset,
        neighbor_link_stat_table_reset,
        pixel_region_table_cleanup,
        pixel_status_table_cleanup,
        pixel_done_table_cleanup,
        neighbor_link_stat_table_cleanup,
    )

cdef class Grid:
    cdef readonly:
        Constants constants
    cdef:
        cGrid _cgrid
    cdef cGrid* to_c(self)
    cpdef void set_values(self, np.ndarray[np.float32_t, ndim=2] fld)
    cpdef void reset(self)
    cdef void reset_tables(self)

cdef cGrid grid_create(np.float32_t[:, :] fld, cConstants constants) except *
cdef void grid_create_pixels(cGrid* grid, np.float32_t[:, :] fld) except *
cdef void grid_set_values(cGrid* grid, np.float32_t[:, :] fld) except *
cdef void grid_reset(cGrid* grid) except *
cdef void grid_cleanup(cGrid* grid) except *
cdef cRegion* grid_new_region(cGrid* grid) except *
cdef cRegions grid_new_regions(cGrid* grid, int n) except *

#==============================================================================
# cRegion
#==============================================================================

cdef int CREGION_NEXT_ID = 0
cdef np.uint64_t cregion_get_unique_id()

cdef void cregion_init(cRegion* cregion, cRegionConf cregion_conf,
        np.uint64_t rid)
cdef void cregion_reset(cRegion* cregion, bint unlink_pixels,
        bint reset_connected)
cdef void cregion_cleanup(cRegion* cregion, bint unlink_pixels,
        bint reset_connected)
cdef void cregion_insert_pixels_coords(cRegion* cregion, cPixel** cpixels,
        np.ndarray[np.int32_t, ndim=2] coords, bint link_region,
        bint unlink_pixels) except *
cdef void cregion_insert_pixel(cRegion* cregion, cPixel* cpixel,
        bint link_region, bint unlink_pixel)
#SR_TMP_NOGIL
cdef void cregion_insert_pixel_nogil(cRegion* cregion, cPixel* cpixel,
        bint link_region, bint unlink_pixel) nogil
cdef void cregion_remove_pixel(cRegion* cregion, cPixel* cpixel)
cdef void cregion_remove_connected(cRegion* cregion, cRegion* cregion_other)
cdef cRegion* cregion_merge(cRegion* cregion1, cRegion* cregion2)
cdef void cregion_reset_boundaries(cRegion* cregion)
cdef void cregion_determine_boundaries(cRegion* cregion, cGrid* grid) except *
cdef bint cregion_overlaps(cRegion* cregion, cRegion* cregion_other)
cdef bint cregion_overlaps_tables(cRegion* cregion, cRegion* cregion_other,
        PixelRegionTable table, PixelRegionTable table_other)
cdef int cregion_overlap_n(cRegion* cregion, cRegion* cregion_other)
cdef int cregion_overlap_n_tables(cRegion* cregion, cRegion* cregion_other,
        PixelRegionTable table, PixelRegionTable table_other)
cdef int cregion_overlap_n_mask(cRegion* cregion,
        np.ndarray[np.uint8_t, ndim=2] mask)

#==============================================================================
# cRegions
#==============================================================================

cdef int CREGIONS_NEXT_ID = 0
cdef np.uint64_t cregions_get_unique_id()

cdef void cregions_init(cRegions* cregions)
cdef cRegions cregions_create(int nmax)
cdef void cregions_reset(cRegions* cregions)
cdef void cregions_cleanup(cRegions* cregions, bint cleanup_regions)
cdef void cregions_link_region(cRegions* cregions, cRegion* cregion,
        bint cleanup, bint unlink_pixels)
cdef void cregions_move(cRegions* source, cRegions* target)
cdef void cregions_connect(cRegion* cregion1, cRegion* cregion2)
cdef void cregions_find_connected(cRegions* cregions, bint reset_existing,
        cConstants* constants) except *
cdef void cregions_determine_boundaries(cRegions* cregions, cGrid* grid) except *

#==============================================================================

cdef void dbg_check_connected(cRegions* cregions, str msg) except *

#==============================================================================

cdef void cpixel_set_region(cPixel* cpixel, cRegion* cregion) nogil

cdef cPixel* cpixel2d_create(
        int n,
    ) nogil
#SR_TMP<
cdef void cpixels_reset(
        cPixel** cpixels,
        np.int32_t nx,
        np.int32_t ny,
    )
#SR_TMP<
cdef cPixel* cpixel_get_neighbor(
        cPixel* cpixel,
        int index,
        cPixel** cpixels,
        np.int32_t nx,
        np.int32_t ny,
        int connectivity,
    )

#==============================================================================
