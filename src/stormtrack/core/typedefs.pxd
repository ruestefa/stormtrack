
# Third-party
cimport numpy as np

# Local
from .structs cimport NeighborLinkStatTable
from .structs cimport PixelDoneTable
from .structs cimport PixelRegionTable
from .structs cimport PixelStatusTable
from .structs cimport cConstants
from .structs cimport cGrid
from .structs cimport cPixel
from .structs cimport cRegion
from .structs cimport cRegionConf
from .structs cimport cRegions
from .structs cimport cRegionsStore
from .structs cimport cregion_conf_default
from .structs cimport cregions_store_create
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
from .tables cimport neighbor_link_stat_table_reset_pixels
from .tables cimport pixel_done_table_cleanup
from .tables cimport pixel_done_table_reset
from .tables cimport pixel_region_table_alloc
from .tables cimport pixel_region_table_cleanup
from .tables cimport pixel_region_table_reset
from .tables cimport pixel_status_table_alloc
from .tables cimport pixel_status_table_cleanup
from .tables cimport pixel_status_table_reset
from .tables cimport pixel_status_table_reset


cpdef Constants default_constants(
    int nx, int ny, int connectivity=?, int n_neighbors_max=?,
)


cdef class Constants:
    cdef readonly:
        int nx
        int ny
        int connectivity
        int n_neighbors_max

    cdef:
        cConstants _cconstants

    cdef cConstants* to_c(self)


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


cdef int CREGION_NEXT_ID = 0


cdef np.uint64_t cregion_get_unique_id()


cdef void cregion_init(cRegion* cregion, cRegionConf cregion_conf, np.uint64_t rid)


cdef void cregion_reset(cRegion* cregion, bint unlink_pixels, bint reset_connected)


cdef void cregion_cleanup(cRegion* cregion, bint unlink_pixels, bint reset_connected)


cdef void cregion_insert_pixels_coords(
    cRegion* cregion,
    cPixel** cpixels,
    np.ndarray[np.int32_t, ndim=2] coords,
    bint link_region,
    bint unlink_pixels,
) except *


cdef void cregion_insert_pixel(
    cRegion* cregion, cPixel* cpixel, bint link_region, bint unlink_pixel,
)


# SR_TMP_NOGIL <<<
cdef void cregion_insert_pixel_nogil(
    cRegion* cregion, cPixel* cpixel, bint link_region, bint unlink_pixel,
) nogil


cdef void cregion_remove_pixel(cRegion* cregion, cPixel* cpixel)


cdef void cregion_remove_connected(cRegion* cregion, cRegion* cregion_other)


cdef cRegion* cregion_merge(cRegion* cregion1, cRegion* cregion2)


cdef void cregion_reset_boundaries(cRegion* cregion)


cdef void cregion_determine_boundaries(cRegion* cregion, cGrid* grid) except *


cdef bint cregion_overlaps(cRegion* cregion, cRegion* cregion_other)


cdef bint cregion_overlaps_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
)


cdef int cregion_overlap_n(cRegion* cregion, cRegion* cregion_other)


cdef int cregion_overlap_n_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
)


cdef int cregion_overlap_n_mask(cRegion* cregion, np.ndarray[np.uint8_t, ndim=2] mask)


cdef int CREGIONS_NEXT_ID = 0


cdef np.uint64_t cregions_get_unique_id()


cdef void cregions_init(cRegions* cregions)


cdef cRegions cregions_create(int nmax)


cdef void cregions_reset(cRegions* cregions)


cdef void cregions_cleanup(cRegions* cregions, bint cleanup_regions)


cdef void cregions_link_region(
    cRegions* cregions, cRegion* cregion, bint cleanup, bint unlink_pixels,
)


cdef void cregions_move(cRegions* source, cRegions* target)


cdef void cregions_connect(cRegion* cregion1, cRegion* cregion2)


cdef void cregions_find_connected(
    cRegions* cregions, bint reset_existing, cConstants* constants,
) except *


cdef void cregions_determine_boundaries(cRegions* cregions, cGrid* grid) except *


cdef void dbg_check_connected(cRegions* cregions, str msg) except *


cdef void cpixel_set_region(cPixel* cpixel, cRegion* cregion) nogil


cdef cPixel* cpixel2d_create(int n) nogil


cdef void cpixels_reset(cPixel** cpixels, np.int32_t nx, np.int32_t ny)  # SR_TMP


cdef cPixel* cpixel_get_neighbor(
        cPixel* cpixel,
        int index,
        cPixel** cpixels,
        np.int32_t nx,
        np.int32_t ny,
        int connectivity,
)
