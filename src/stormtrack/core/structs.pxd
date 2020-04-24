
# Third-party
cimport cython
cimport numpy as np


# cConstants
# pixeltype
# cPixel
# cField2D
# cRegionConf
# cregion_conf_default
# cRegion
# cRegions
# get_matching_neighbor_id
# cRegionRankSlot
# cRegionRankSlots
# cRegionsStore
# cregions_store_create
# cGrid
# grid_create_empty
# SuccessorCandidate
# SuccessorCandidates


# :call: > --- CALLERS ---
# :call: > core::structs::cGrid
# :call: > core::structs::grid_create_empty
# :call: v --- CALLING ---
# :call: v core::identification::_cregions_merge_connected_core
# :call: v core::identification::_find_features_threshold_random_seeds
# :call: v core::identification::c_find_features_2d_threshold_seeds
# :call: v core::identification::cregions2features_connected2neighbors
# :call: v core::identification::cregions_create_features
# :call: v core::identification::cregions_merge_connected
# :call: v core::identification::cregions_merge_connected_inplace
# :call: v core::identification::csplit_regiongrow_levels
# :call: v core::identification::feature_split_regiongrow
# :call: v core::identification::feature_to_cregion
# :call: v core::identification::features_find_neighbors
# :call: v core::identification::features_find_neighbors_core
# :call: v core::identification::features_grow
# :call: v core::identification::features_to_cregions
# :call: v core::identification::find_features_2d_threshold
# :call: v core::identification::find_features_2d_threshold_seeded
# :call: v core::identification::merge_adjacent_features
# :call: v core::identification::pixels_find_boundaries
# :call: v core::tables::neighbor_link_stat_table_alloc
# :call: v core::tables::neighbor_link_stat_table_init
# :call: v core::tables::neighbor_link_stat_table_reset
# :call: v core::tables::pixel_done_table_alloc
# :call: v core::tables::pixel_done_table_cleanup
# :call: v core::tables::pixel_region_table_alloc
# :call: v core::tables::pixel_region_table_alloc_grid
# :call: v core::tables::pixel_status_table_alloc
# :call: v core::typedefs::Constants::__cinit__
# :call: v core::typedefs::Constants::to_c
# :call: v core::typedefs::_collect_neighbors
# :call: v core::typedefs::cregions_find_connected
# :call: v core::typedefs::grid_create
cdef struct cConstants:
    np.int32_t nx
    np.int32_t ny
    np.uint8_t connectivity
    np.uint8_t n_neighbors_max # SR_TODO rename to pixel_neighbors_max


# :call: > --- CALLERS ---
# :call: > core::structs::cPixel
# :call: v --- CALLING ---
# :call: v core::identification::grow_cregion_rec
# :call: v core::typedefs::cpixel2d_create
# :call: v core::typedefs::cpixels_reset
cdef enum pixeltype:
    pixeltype_none
    pixeltype_background
    pixeltype_feature


# :call: > --- CALLERS ---
# :call: > core::identification::_cregions_merge_connected_core
# :call: > core::identification::_find_background_neighbor_pixels
# :call: > core::identification::assign_cpixel
# :call: > core::identification::c_find_features_2d_threshold_seeds_core
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::collect_adjacent_pixels
# :call: > core::identification::collect_pixels
# :call: > core::identification::cpixel2arr
# :call: > core::identification::cpixel_count_neighbors_in_cregion
# :call: > core::identification::create_feature
# :call: > core::identification::cregions_merge_connected
# :call: > core::identification::csplit_regiongrow_core
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::identification::csplit_regiongrow_levels_core
# :call: > core::identification::determine_shared_boundary_pixels
# :call: > core::identification::extract_subregions_level
# :call: > core::identification::feature_split_regiongrow
# :call: > core::identification::find_features_2d_threshold
# :call: > core::identification::grow_cregion_rec
# :call: > core::identification::pop_random_unassigned_pixel
# :call: > core::identification::regiongrow_advance_boundary
# :call: > core::identification::regiongrow_assign_pixel
# :call: > core::identification::regiongrow_resolve_multi_assignments
# :call: > core::identification::resolve_multi_assignment
# :call: > core::identification::resolve_multi_assignment_best_connected_region
# :call: > core::identification::resolve_multi_assignment_biggest_region
# :call: > core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > core::identification::resolve_multi_assignment_strongest_region
# :call: > core::structs::cField2D
# :call: > core::structs::cGrid
# :call: > core::structs::cRegion
# :call: > core::tables::neighbor_link_stat_table_init
# :call: > core::tables::pixel_done_table_init
# :call: > core::tables::pixel_done_table_reset
# :call: > core::tables::pixel_region_table_init_regions
# :call: > core::tables::pixel_status_table_init_feature
# :call: > core::typedefs::_collect_neighbors
# :call: > core::typedefs::_cpixel_get_neighbor
# :call: > core::typedefs::_cpixel_unlink_region
# :call: > core::typedefs::_cregion_create_pixels
# :call: > core::typedefs::_cregion_determine_boundaries_core
# :call: > core::typedefs::_cregion_extend_hole
# :call: > core::typedefs::_cregion_extend_holes
# :call: > core::typedefs::_cregion_extend_pixels
# :call: > core::typedefs::_cregion_extend_pixels_nogil
# :call: > core::typedefs::_cregion_extend_shell
# :call: > core::typedefs::_cregion_extend_shells
# :call: > core::typedefs::_cregion_insert_hole_pixel
# :call: > core::typedefs::_cregion_insert_shell_pixel
# :call: > core::typedefs::_cregion_overlap_core
# :call: > core::typedefs::_cregion_reconnect_pixel
# :call: > core::typedefs::_cregion_remove_pixel_from_holes
# :call: > core::typedefs::_cregion_remove_pixel_from_holes_nogil
# :call: > core::typedefs::_cregion_remove_pixel_from_pixels
# :call: > core::typedefs::_cregion_remove_pixel_from_pixels_nogil
# :call: > core::typedefs::_cregion_remove_pixel_from_shells
# :call: > core::typedefs::_cregion_remove_pixel_from_shells_nogil
# :call: > core::typedefs::_determine_boundary_pixels_raw
# :call: > core::typedefs::_extract_closed_path
# :call: > core::typedefs::_find_link_to_continue
# :call: > core::typedefs::_reconstruct_boundaries
# :call: > core::typedefs::categorize_boundaries
# :call: > core::typedefs::cpixel2d_create
# :call: > core::typedefs::cpixel_get_neighbor
# :call: > core::typedefs::cpixel_set_region
# :call: > core::typedefs::cpixels_reset
# :call: > core::typedefs::cregion_check_validity
# :call: > core::typedefs::cregion_determine_bbox
# :call: > core::typedefs::cregion_init
# :call: > core::typedefs::cregion_insert_pixel
# :call: > core::typedefs::cregion_insert_pixel_nogil
# :call: > core::typedefs::cregion_insert_pixels_coords
# :call: > core::typedefs::cregion_northernmost_pixel
# :call: > core::typedefs::cregion_overlap_n_mask
# :call: > core::typedefs::cregion_remove_pixel
# :call: > core::typedefs::cregion_remove_pixel_nogil
# :call: > core::typedefs::cregion_reset
# :call: > core::typedefs::cregions_find_connected
# :call: > core::typedefs::cregions_find_northernmost_uncategorized_region
# :call: > core::typedefs::grid_create_pixels
# :call: > core::typedefs::grid_set_values
# :call: > core::typedefs::neighbor_pixel_angle
# :call: v --- CALLING ---
# :call: v core::structs::cRegion
# :call: v core::structs::pixeltype
cdef struct cPixel:
    np.uint64_t id
    # SR_TODO merge x, y and xy
    np.int32_t x
    np.int32_t y
    np.float32_t v
    pixeltype type
    cPixel* neighbors[8]
    np.uint8_t neighbors_max
    np.uint8_t neighbors_n
    np.uint8_t connectivity
    cRegion* region
    bint is_seed
    bint is_feature_boundary


# :call: > --- CALLERS ---
# :call: v --- CALLING ---
# :call: v core::structs::cPixel
cdef struct cField2D:
    cPixel** pixels
    np.int32_t nx
    np.int32_t ny


# :call: > --- CALLERS ---
# :call: > core::identification::_find_features_threshold_random_seeds
# :call: > core::identification::c_find_features_2d_threshold_seeds
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::cregions_merge_connected
# :call: > core::identification::cregions_merge_connected_inplace
# :call: > core::identification::feature_to_cregion
# :call: > core::identification::features_to_cregions
# :call: > core::identification::find_features_2d_threshold
# :call: > core::identification::pixels_find_boundaries
# :call: > core::structs::cregion_conf_default
# :call: > core::typedefs::cregion_init
# :call: > core::typedefs::cregions_store_extend
# :call: v --- CALLING ---
cdef struct cRegionConf:
    int connected_max
    int pixels_max
    int shells_max
    int shell_max
    int holes_max
    int hole_max


# :call: > --- CALLERS ---
# :call: > core::identification::collect_adjacent_pixels
# :call: > core::identification::cregions2features_connected2neighbors
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::identification::determine_shared_boundary_pixels
# :call: > core::identification::extract_subregions_level
# :call: > core::identification::feature_split_regiongrow
# :call: > core::identification::features_find_neighbors_core
# :call: > core::identification::features_grow
# :call: > core::identification::merge_adjacent_features
# :call: > core::identification::regiongrow_advance_boundary
# :call: > core::typedefs::_determine_boundary_pixels_raw
# :call: > core::typedefs::cregions_store_extend
# :call: v --- CALLING ---
# :call: v core::structs::cRegionConf
cdef inline cRegionConf cregion_conf_default():
    cdef cRegionConf conf = cRegionConf(
            connected_max = 20,
            pixels_max = 200,
            shells_max = 1,
            shell_max = 20,
            holes_max = 5,
            hole_max = 20,
        )
    return conf


# :call: > --- CALLERS ---
# :call: > core::identification::Feature::set_cregion
# :call: > core::identification::_cregion_collect_connected_regions_rec
# :call: > core::identification::_cregions_merge_connected_core
# :call: > core::identification::_find_background_neighbor_pixels
# :call: > core::identification::assert_no_unambiguously_assigned_pixels
# :call: > core::identification::assign_cpixel
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::collect_adjacent_pixels
# :call: > core::identification::collect_pixels
# :call: > core::identification::cpixel_count_neighbors_in_cregion
# :call: > core::identification::cregion_collect_connected_regions
# :call: > core::identification::cregion_find_corresponding_feature
# :call: > core::identification::cregions2features_connected2neighbors
# :call: > core::identification::cregions_create_features
# :call: > core::identification::cregions_merge_connected
# :call: > core::identification::csplit_regiongrow_core
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::identification::csplit_regiongrow_levels_core
# :call: > core::identification::dbg_print_selected_regions
# :call: > core::identification::determine_shared_boundary_pixels
# :call: > core::identification::eliminate_regions_by_size
# :call: > core::identification::extract_subregions_level
# :call: > core::identification::feature_split_regiongrow
# :call: > core::identification::feature_to_cregion
# :call: > core::identification::features_grow
# :call: > core::identification::features_neighbors_to_cregions_connected
# :call: > core::identification::features_to_cregions
# :call: > core::identification::find_existing_region
# :call: > core::identification::find_features_2d_threshold
# :call: > core::identification::initialize_surrounding_background_region
# :call: > core::identification::pixels_find_boundaries
# :call: > core::identification::regiongrow_advance_boundary
# :call: > core::identification::regiongrow_assign_pixel
# :call: > core::identification::regiongrow_resolve_multi_assignments
# :call: > core::identification::resolve_multi_assignment_biggest_region
# :call: > core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > core::identification::resolve_multi_assignment_strongest_region
# :call: > core::structs::SuccessorCandidate
# :call: > core::structs::cPixel
# :call: > core::structs::cRegions
# :call: > core::structs::cRegionsStore
# :call: > core::tables::cregion_rank_slots_insert_region
# :call: > core::tables::neighbor_link_stat_table_init
# :call: > core::tables::neighbor_link_stat_table_reset_pixels
# :call: > core::tables::pixel_done_table_init
# :call: > core::tables::pixel_done_table_reset
# :call: > core::tables::pixel_region_table_alloc_pixels
# :call: > core::tables::pixel_region_table_cleanup_pixels
# :call: > core::tables::pixel_region_table_grow
# :call: > core::tables::pixel_region_table_init_regions
# :call: > core::tables::pixel_region_table_insert_region
# :call: > core::tables::pixel_region_table_reset_region
# :call: > core::tables::pixel_status_table_init_feature
# :call: > core::tables::pixel_status_table_reset_feature
# :call: > core::typedefs::_cpixel_unlink_region
# :call: > core::typedefs::_cregion_add_connected
# :call: > core::typedefs::_cregion_create_pixels
# :call: > core::typedefs::_cregion_determine_boundaries_core
# :call: > core::typedefs::_cregion_extend_hole
# :call: > core::typedefs::_cregion_extend_holes
# :call: > core::typedefs::_cregion_extend_pixels
# :call: > core::typedefs::_cregion_extend_pixels_nogil
# :call: > core::typedefs::_cregion_extend_shell
# :call: > core::typedefs::_cregion_extend_shells
# :call: > core::typedefs::_cregion_hole_remove_gaps
# :call: > core::typedefs::_cregion_hole_remove_gaps_nogil
# :call: > core::typedefs::_cregion_insert_hole_pixel
# :call: > core::typedefs::_cregion_insert_shell_pixel
# :call: > core::typedefs::_cregion_new_hole
# :call: > core::typedefs::_cregion_new_shell
# :call: > core::typedefs::_cregion_overlap_core
# :call: > core::typedefs::_cregion_reconnect_pixel
# :call: > core::typedefs::_cregion_remove_pixel_from_holes
# :call: > core::typedefs::_cregion_remove_pixel_from_holes_nogil
# :call: > core::typedefs::_cregion_remove_pixel_from_pixels
# :call: > core::typedefs::_cregion_remove_pixel_from_pixels_nogil
# :call: > core::typedefs::_cregion_remove_pixel_from_shells
# :call: > core::typedefs::_cregion_remove_pixel_from_shells_nogil
# :call: > core::typedefs::_cregion_reset_connected
# :call: > core::typedefs::_cregion_shell_remove_gaps
# :call: > core::typedefs::_cregion_shell_remove_gaps_nogil
# :call: > core::typedefs::_determine_boundary_pixels_raw
# :call: > core::typedefs::_extract_closed_path
# :call: > core::typedefs::_find_link_to_continue
# :call: > core::typedefs::_reconstruct_boundaries
# :call: > core::typedefs::categorize_boundaries
# :call: > core::typedefs::cpixel_set_region
# :call: > core::typedefs::cregion_check_validity
# :call: > core::typedefs::cregion_cleanup
# :call: > core::typedefs::cregion_determine_bbox
# :call: > core::typedefs::cregion_determine_boundaries
# :call: > core::typedefs::cregion_init
# :call: > core::typedefs::cregion_insert_pixel
# :call: > core::typedefs::cregion_insert_pixel_nogil
# :call: > core::typedefs::cregion_insert_pixels_coords
# :call: > core::typedefs::cregion_merge
# :call: > core::typedefs::cregion_northernmost_pixel
# :call: > core::typedefs::cregion_overlap_n
# :call: > core::typedefs::cregion_overlap_n_mask
# :call: > core::typedefs::cregion_overlap_n_tables
# :call: > core::typedefs::cregion_overlaps
# :call: > core::typedefs::cregion_overlaps_tables
# :call: > core::typedefs::cregion_pixels_remove_gaps
# :call: > core::typedefs::cregion_pixels_remove_gaps_nogil
# :call: > core::typedefs::cregion_remove_connected
# :call: > core::typedefs::cregion_remove_pixel
# :call: > core::typedefs::cregion_remove_pixel_nogil
# :call: > core::typedefs::cregion_reset
# :call: > core::typedefs::cregion_reset_boundaries
# :call: > core::typedefs::cregions_connect
# :call: > core::typedefs::cregions_create
# :call: > core::typedefs::cregions_determine_boundaries
# :call: > core::typedefs::cregions_extend
# :call: > core::typedefs::cregions_find_connected
# :call: > core::typedefs::cregions_link_region
# :call: > core::typedefs::cregions_store_extend
# :call: > core::typedefs::cregions_store_get_new_region
# :call: > core::typedefs::dbg_check_connected
# :call: > core::typedefs::grid_new_region
# :call: > core::typedefs::grid_new_regions
# :call: v --- CALLING ---
# :call: v core::structs::cPixel
cdef struct cRegion:
    np.uint64_t id
    cRegion** connected
    int connected_n
    int connected_max
    cPixel** pixels
    int pixels_n
    int pixels_istart
    int pixels_iend
    int pixels_max
    cPixel*** shells
    int shells_n
    int shells_max
    int* shell_n
    int* shell_max
    cPixel*** holes
    int holes_n
    int holes_max
    int* hole_n
    int* hole_max


# :call: > --- CALLERS ---
# :call: > core::identification::_cregion_collect_connected_regions_rec
# :call: > core::identification::_cregions_merge_connected_core
# :call: > core::identification::_find_features_threshold_random_seeds
# :call: > core::identification::assign_cpixel
# :call: > core::identification::c_find_features_2d_threshold_seeds
# :call: > core::identification::c_find_features_2d_threshold_seeds_core
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::collect_pixels
# :call: > core::identification::cregion_collect_connected_regions
# :call: > core::identification::cregions2features_connected2neighbors
# :call: > core::identification::cregions_create_features
# :call: > core::identification::cregions_merge_connected
# :call: > core::identification::cregions_merge_connected_inplace
# :call: > core::identification::csplit_regiongrow_core
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::identification::csplit_regiongrow_levels_core
# :call: > core::identification::eliminate_regions_by_size
# :call: > core::identification::extract_subregions_level
# :call: > core::identification::feature_split_regiongrow
# :call: > core::identification::feature_to_cregion
# :call: > core::identification::features_find_neighbors_core
# :call: > core::identification::features_grow
# :call: > core::identification::features_neighbors_to_cregions_connected
# :call: > core::identification::features_to_cregions
# :call: > core::identification::find_features_2d_threshold
# :call: > core::identification::grow_cregion_rec
# :call: > core::identification::initialize_surrounding_background_region
# :call: > core::identification::merge_adjacent_features
# :call: > core::identification::regiongrow_advance_boundary
# :call: > core::tables::pixel_region_table_init_regions
# :call: > core::tables::pixel_region_table_reset_regions
# :call: > core::tables::pixel_status_table_init_feature
# :call: > core::typedefs::_cregion_determine_boundaries_core
# :call: > core::typedefs::_reconstruct_boundaries
# :call: > core::typedefs::categorize_boundaries
# :call: > core::typedefs::cregion_determine_boundaries
# :call: > core::typedefs::cregions_cleanup
# :call: > core::typedefs::cregions_create
# :call: > core::typedefs::cregions_determine_boundaries
# :call: > core::typedefs::cregions_extend
# :call: > core::typedefs::cregions_find_connected
# :call: > core::typedefs::cregions_find_northernmost_uncategorized_region
# :call: > core::typedefs::cregions_init
# :call: > core::typedefs::cregions_link_region
# :call: > core::typedefs::cregions_move
# :call: > core::typedefs::cregions_reset
# :call: > core::typedefs::dbg_check_connected
# :call: > core::typedefs::grid_new_regions
# :call: v --- CALLING ---
# :call: v core::structs::cRegion
cdef struct cRegions:
    cRegion** regions
    int n
    int max
    np.uint64_t id


# :call: > --- CALLERS ---
# :call: > core::tables::neighbor_link_stat_table_init
# :call: > core::typedefs::_reconstruct_boundaries
# :call: v --- CALLING ---
@cython.cdivision
cdef inline np.uint8_t get_matching_neighbor_id(np.uint8_t ind, int nmax) nogil:
    cdef int mind = (ind + nmax / 2) % nmax
    return mind


# :call: > --- CALLERS ---
# :call: > core::identification::regiongrow_resolve_multi_assignments
# :call: > core::identification::resolve_multi_assignment
# :call: > core::tables::cregion_rank_slots_copy
# :call: > core::tables::cregion_rank_slots_extend
# :call: > core::tables::pixel_region_table_alloc
# :call: > core::tables::pixel_region_table_alloc_pixel
# :call: v --- CALLING ---
cdef struct cRegionRankSlot:
    cRegion* region
    np.int8_t rank


# :call: > --- CALLERS ---
# :call: > core::identification::dbg_print_selected_regions
# :call: > core::identification::regiongrow_resolve_multi_assignments
# :call: > core::identification::resolve_multi_assignment
# :call: > core::identification::resolve_multi_assignment_best_connected_region
# :call: > core::identification::resolve_multi_assignment_biggest_region
# :call: > core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > core::identification::resolve_multi_assignment_strongest_region
# :call: > core::tables::cregion_rank_slots_copy
# :call: > core::tables::cregion_rank_slots_extend
# :call: > core::tables::cregion_rank_slots_insert_region
# :call: > core::tables::pixel_region_table_alloc
# :call: > core::tables::pixel_region_table_alloc_grid
# :call: v --- CALLING ---
cdef struct cRegionRankSlots:
    cRegionRankSlot* slots
    int max
    int n


# :call: > --- CALLERS ---
# :call: > core::tables::_pixel_region_table_cleanup_entry
# :call: > core::tables::pixel_region_table_alloc
# :call: > core::tables::pixel_region_table_alloc_grid
# :call: > core::tables::pixel_region_table_alloc_pixel
# :call: > core::tables::pixel_region_table_alloc_pixels
# :call: > core::tables::pixel_region_table_cleanup
# :call: > core::tables::pixel_region_table_cleanup_pixels
# :call: > core::tables::pixel_region_table_grow
# :call: > core::tables::pixel_region_table_init_regions
# :call: > core::tables::pixel_region_table_insert_region
# :call: > core::tables::pixel_region_table_reset
# :call: > core::tables::pixel_region_table_reset_region
# :call: > core::tables::pixel_region_table_reset_regions
# :call: > core::tables::pixel_region_table_reset_slots
# :call: > core::typedefs::_cregion_overlap_core
# :call: > core::typedefs::cregion_overlap_n_tables
# :call: > core::typedefs::cregion_overlaps_tables
# :call: v --- CALLING ---
ctypedef cRegionRankSlots** PixelRegionTable


# :call: > --- CALLERS ---
# :call: > core::tables::pixel_status_table_alloc
# :call: > core::tables::pixel_status_table_cleanup
# :call: > core::tables::pixel_status_table_init_feature
# :call: > core::tables::pixel_status_table_reset
# :call: > core::tables::pixel_status_table_reset_feature
# :call: v --- CALLING ---
ctypedef np.int8_t** PixelStatusTable


# :call: > --- CALLERS ---
# :call: > core::tables::pixel_done_table_alloc
# :call: > core::tables::pixel_done_table_cleanup
# :call: > core::tables::pixel_done_table_init
# :call: > core::tables::pixel_done_table_reset
# :call: v --- CALLING ---
ctypedef bint **PixelDoneTable


# :call: > --- CALLERS ---
# :call: > core::tables::neighbor_link_stat_table_alloc
# :call: > core::tables::neighbor_link_stat_table_reset
# :call: > core::tables::neighbor_link_stat_table_reset_pixels
# :call: > core::tables::neighbor_link_stat_table_cleanup
# :call: > core::tables::neighbor_link_stat_table_init
# :call: v --- CALLING ---
ctypedef np.int8_t ***NeighborLinkStatTable


# :call: > --- CALLERS ---
# :call: > core::structs::cGrid
# :call: > core::structs::cregions_store_create
# :call: > core::typedefs::cregions_store_cleanup
# :call: > core::typedefs::cregions_store_extend
# :call: > core::typedefs::cregions_store_get_new_region
# :call: > core::typedefs::cregions_store_reset
# :call: v --- CALLING ---
# :call: v core::structs::cRegion
cdef struct cRegionsStore:
    cRegion** blocks
    int block_size
    int i_block
    int n_blocks
    int i_next_region


# :call: > --- CALLERS ---
# :call: > core::structs::grid_create_empty
# :call: v --- CALLING ---
# :call: v core::structs::cRegionsStore
cdef inline cRegionsStore cregions_store_create():
    return cRegionsStore(
        blocks= NULL,
        block_size= 50,
        i_block= 0,
        n_blocks= 0,
        i_next_region= 0,
        )


# :call: > --- CALLERS ---
# :call: > core::identification::_cregions_merge_connected_core
# :call: > core::identification::_find_features_threshold_random_seeds
# :call: > core::identification::assign_cpixel
# :call: > core::identification::c_find_features_2d_threshold_seeds
# :call: > core::identification::c_find_features_2d_threshold_seeds_core
# :call: > core::identification::cfeatures_grow_core
# :call: > core::identification::collect_adjacent_pixels
# :call: > core::identification::collect_pixels
# :call: > core::identification::cpixel_count_neighbors_in_cregion
# :call: > core::identification::cregions2features_connected2neighbors
# :call: > core::identification::cregions_create_features
# :call: > core::identification::cregions_merge_connected
# :call: > core::identification::cregions_merge_connected_inplace
# :call: > core::identification::csplit_regiongrow_core
# :call: > core::identification::csplit_regiongrow_levels
# :call: > core::identification::csplit_regiongrow_levels_core
# :call: > core::identification::determine_shared_boundary_pixels
# :call: > core::identification::extract_subregions_level
# :call: > core::identification::feature_split_regiongrow
# :call: > core::identification::feature_to_cregion
# :call: > core::identification::features_find_neighbors
# :call: > core::identification::features_find_neighbors_core
# :call: > core::identification::features_grow
# :call: > core::identification::features_to_cregions
# :call: > core::identification::find_existing_region
# :call: > core::identification::find_features_2d_threshold
# :call: > core::identification::grow_cregion_rec
# :call: > core::identification::init_random_seeds
# :call: > core::identification::initialize_surrounding_background_region
# :call: > core::identification::merge_adjacent_features
# :call: > core::identification::pixels_find_boundaries
# :call: > core::identification::pop_random_unassigned_pixel
# :call: > core::identification::regiongrow_advance_boundary
# :call: > core::identification::regiongrow_assign_pixel
# :call: > core::identification::regiongrow_resolve_multi_assignments
# :call: > core::identification::resolve_multi_assignment
# :call: > core::identification::resolve_multi_assignment_best_connected_region
# :call: > core::structs::grid_create_empty
# :call: > core::typedefs::Grid::to_c
# :call: > core::typedefs::_cregion_determine_boundaries_core
# :call: > core::typedefs::_determine_boundary_pixels_raw
# :call: > core::typedefs::_reconstruct_boundaries
# :call: > core::typedefs::boundary_must_be_a_shell
# :call: > core::typedefs::categorize_boundaries
# :call: > core::typedefs::cregion_determine_boundaries
# :call: > core::typedefs::cregions_determine_boundaries
# :call: > core::typedefs::grid_cleanup
# :call: > core::typedefs::grid_create
# :call: > core::typedefs::grid_create_pixels
# :call: > core::typedefs::grid_new_region
# :call: > core::typedefs::grid_new_regions
# :call: > core::typedefs::grid_reset
# :call: > core::typedefs::grid_set_values
# :call: v --- CALLING ---
# :call: v core::structs::cConstants
# :call: v core::structs::cPixel
# :call: v core::structs::cRegionsStore
cdef struct cGrid:
    np.uint64_t timestep
    cConstants constants
    cPixel** pixels
    PixelRegionTable pixel_region_table
    PixelStatusTable pixel_status_table
    PixelDoneTable pixel_done_table
    NeighborLinkStatTable neighbor_link_stat_table
    cRegionsStore _regions


# :call: > --- CALLERS ---
# :call: > core::typedefs::grid_create
# :call: v --- CALLING ---
# :call: v core::structs::cConstants
# :call: v core::structs::cGrid
# :call: v core::structs::cregions_store_create
cdef inline cGrid grid_create_empty(cConstants constants):
    return cGrid(
        timestep=0,
        constants=constants,
        pixels=NULL,
        constants=constants,
        pixel_region_table=NULL,
        pixel_status_table=NULL,
        pixel_done_table=NULL,
        neighbor_link_stat_table=NULL,
        _regions=cregions_store_create(),
        )


# :call: > --- CALLERS ---
# :call: > core::structs::SuccessorCandidates
# :call: v --- CALLING ---
# :call: v core::structs::cRegion
cdef struct SuccessorCandidate:
    cRegion*      parent
    cRegion**      children
    np.float32_t *p_shares
    np.int32_t   *n_overlaps
    int n
    int max
    int direction
    float p_tot
    float p_size
    float p_overlap


# :call: > --- CALLERS ---
# :call: v --- CALLING ---
# :call: v core::structs::SuccessorCandidate
cdef struct SuccessorCandidates:
    SuccessorCandidate* candidates
    int n
    int max
