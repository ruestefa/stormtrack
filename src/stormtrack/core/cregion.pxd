# -*- coding: utf-8 -*-

# Third-party
cimport numpy as np

# Local
from .structs cimport PixelRegionTable
from .structs cimport cConstants
from .structs cimport cGrid
from .structs cimport cPixel
from .structs cimport cRegion
from .structs cimport cRegionConf
from .structs cimport cregion_conf_default


cdef int CREGION_NEXT_ID = 0


# SR_TMP <
cdef inline void _cregion_remove_pixel_from_pixels(cRegion* cregion, cPixel* cpixel)
cdef inline void _cregion_remove_pixel_from_pixels_nogil(cRegion* cregion, cPixel* cpixel) nogil
cdef inline void _cregion_remove_pixel_from_shells(cRegion* cregion, cPixel* cpixel)
cdef inline void _cregion_remove_pixel_from_shells_nogil(cRegion* cregion, cPixel* cpixel) nogil
cdef inline void _cregion_remove_pixel_from_holes(cRegion* cregion, cPixel* cpixel)
cdef inline void _cregion_remove_pixel_from_holes_nogil(cRegion* cregion, cPixel* cpixel) nogil
cdef void _cregion_extend_pixels(cRegion* cregion)
cdef void _cregion_extend_pixels_nogil(cRegion* cregion) nogil
cdef void _cregion_extend_shell(cRegion* cregion, int i_shell)
cdef void _cregion_extend_hole(cRegion* cregion, int i_hole)
cdef void _cregion_extend_shells(cRegion* cregion)
cdef void _cregion_extend_holes(cRegion* cregion)
cdef void _cregion_new_shell(cRegion* cregion)
cdef void _cregion_new_hole(cRegion* cregion)
cdef void _cregion_add_connected(cRegion* cregion, cRegion* cregion_other)
cdef void _cregion_extend_connected(cRegion* cregion)
cdef void cregion_remove_connected(cRegion* cregion, cRegion* cregion_other)
cdef bint _find_link_to_continue(cPixel** cpixel, np.uint8_t *i_neighbor, cRegion* boundary_pixels, np.int8_t ***neighbor_link_stat_table)
cdef bint _extract_closed_path(cRegion* boundary)
cdef void cregion_check_validity(cRegion* cregion, int idx) except *
cdef cPixel* cregion_northernmost_pixel(cRegion* cregion)
cdef inline void _cregion_reset_connected(cRegion* cregion, bint unlink)
cdef void cregion_reset_boundaries(cRegion* cregion)
cdef inline int _collect_neighbors(np.int32_t i, np.int32_t j, cPixel** neighbors, cPixel** cpixels, cConstants* constants, int connectivity)
cdef void cregion_determine_bbox(cRegion* cregion, cPixel* lower_left, cPixel* upper_right)
cdef void cregion_remove_pixel_nogil(cRegion* cregion, cPixel* cpixel) nogil
cdef void _cregion_reconnect_pixel(cRegion* cregion, cPixel* cpixel, bint warn)
cdef void _cregion_insert_shell_pixel(cRegion* cregion, int i_shell, cPixel* cpixel)
cdef void _cregion_insert_hole_pixel(cRegion* cregion, int i_hole, cPixel* cpixel)
cdef inline void _cpixel_unlink_region(cPixel* cpixel, cRegion* cregion) nogil
cdef cRegion _determine_boundary_pixels_raw(cRegion* cregion, cGrid* grid)
# SR_TMP >


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::cregions_store::cregions_store_extend
cdef np.uint64_t cregion_get_unique_id()


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::cregions_store::cregions_store_extend
cdef void cregion_init(cRegion* cregion, cRegionConf cregion_conf, np.uint64_t rid)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cpixel_count_neighbors_in_cregion
cdef cPixel* cpixel_get_neighbor(
        cPixel* cpixel,
        int index,
        cPixel** cpixels,
        np.int32_t nx,
        np.int32_t ny,
        int connectivity,
)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
cdef int cregion_overlap_n_mask(cRegion* cregion, np.ndarray[np.uint8_t, ndim=2] mask)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::cregion::_cpixel_unlink_region
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion::_cregion_reconnect_pixel
# :call: > stormtrack::core::cregion::cregion_insert_pixel
# :call: > stormtrack::core::cregion::cregion_insert_pixel_nogil
cdef void cpixel_set_region(cPixel* cpixel, cRegion* cregion) nogil


# :call: > --- callers ---
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::cregion::cregion_insert_pixel
cdef void cregion_remove_pixel(cRegion* cregion, cPixel* cpixel)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::cregion::cregion_cleanup
# :call: > stormtrack::core::cregions::cregions_reset
# :call: > stormtrack::core::cregions_store::cregions_store_reset
cdef void cregion_reset(cRegion* cregion, bint unlink_pixels, bint reset_connected)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::Feature::cleanup_cregion
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::eliminate_regions_by_size
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion::cregion_merge
# :call: > stormtrack::core::cregions::cregions_cleanup
# :call: > stormtrack::core::cregions::cregions_link_region
# :call: > stormtrack::core::cregions_store::cregions_store_cleanup
cdef void cregion_cleanup(cRegion* cregion, bint unlink_pixels, bint reset_connected)


# :call: > --- callers ---
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
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion::cregion_insert_pixels_coords
# :call: > stormtrack::core::cregion::cregion_merge
cdef void cregion_insert_pixel(
    cRegion* cregion, cPixel* cpixel, bint link_region, bint unlink_pixel,
)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_background_neighbor_pixels
# SR_TMP_NOGIL <<<
cdef void cregion_insert_pixel_nogil(
    cRegion* cregion, cPixel* cpixel, bint link_region, bint unlink_pixel,
) nogil


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::pixels_find_boundaries
cdef void cregion_insert_pixels_coords(
    cRegion* cregion,
    cPixel** cpixels,
    np.ndarray[np.int32_t, ndim=2] coords,
    bint link_region,
    bint unlink_pixels,
) except *


# :call: > --- callers ---
# :call: > stormtrack::core::identification::find_existing_region
cdef cRegion* cregion_merge(cRegion* cregion1, cRegion* cregion2)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: > stormtrack::core::identification::grow_cregion_rec
# :call: > stormtrack::core::cregions::cregions_find_connected
cdef void cregion_connect_with(cRegion* cregion1, cRegion* cregion2)


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::FeatureTracker::_extend_tracks_core
cdef bint cregion_overlaps_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
)


# :call: > --- callers ---
# :call: > stormtrack::extra::front_surgery::*
cdef int cregion_overlap_n(cRegion* cregion, cRegion* cregion_other)


# :call: > --- callers ---
# :call: > stormtrack::core::tracking::FeatureTracker::_compute_successor_probabilities
cdef int cregion_overlap_n_tables(
    cRegion* cregion,
    cRegion* cregion_other,
    PixelRegionTable table,
    PixelRegionTable table_other,
)
