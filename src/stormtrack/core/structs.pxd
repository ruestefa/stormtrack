
# Third-party
cimport cython
cimport numpy as np


# :call: > --- callers ---
# :call: > stormtrack::core::structs::cGrid
# :call: > stormtrack::core::structs::grid_create_empty
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::identification::_cregions_merge_connected_core
# :call: v stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: v stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: v stormtrack::core::identification::cregions2features_connected2neighbors
# :call: v stormtrack::core::identification::cregions_create_features
# :call: v stormtrack::core::identification::cregions_merge_connected
# :call: v stormtrack::core::identification::cregions_merge_connected_inplace
# :call: v stormtrack::core::identification::csplit_regiongrow_levels
# :call: v stormtrack::core::identification::feature_split_regiongrow
# :call: v stormtrack::core::identification::feature_to_cregion
# :call: v stormtrack::core::identification::features_find_neighbors
# :call: v stormtrack::core::identification::features_find_neighbors_core
# :call: v stormtrack::core::identification::features_grow
# :call: v stormtrack::core::identification::features_to_cregions
# :call: v stormtrack::core::identification::find_features_2d_threshold
# :call: v stormtrack::core::identification::find_features_2d_threshold_seeded
# :call: v stormtrack::core::identification::merge_adjacent_features
# :call: v stormtrack::core::identification::pixels_find_boundaries
# :call: v stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: v stormtrack::core::tables::neighbor_link_stat_table_init
# :call: v stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: v stormtrack::core::tables::pixel_done_table_alloc
# :call: v stormtrack::core::tables::pixel_done_table_cleanup
# :call: v stormtrack::core::tables::pixel_region_table_alloc
# :call: v stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v stormtrack::core::tables::pixel_status_table_alloc
# :call: v stormtrack::core::constants::Constants::__cinit__
# :call: v stormtrack::core::constants::Constants::to_c
# :call: v stormtrack::core::cregion::_collect_neighbors
# :call: v stormtrack::core::cregions::cregions_find_connected
# :call: v stormtrack::core::grid::grid_create
cdef struct cConstants:
    np.int32_t nx
    np.int32_t ny
    np.uint8_t connectivity
    np.uint8_t n_neighbors_max # SR_TODO rename to pixel_neighbors_max


# :call: > --- callers ---
# :call: > stormtrack::core::structs::cPixel
# :call: v --- calling ---
# :call: v stormtrack::core::identification::grow_cregion_rec
# :call: v stormtrack::core::cpixel::cpixel2d_create
# :call: v stormtrack::core::cpixel::cpixels_reset
cdef enum pixeltype:
    pixeltype_none
    pixeltype_background
    pixeltype_feature


# :call: > --- callers ---
# :call: > stormtrack::core::cpixel::cpixel2d_create
# :call: > stormtrack::core::cpixel::cpixels_reset
# :call: > stormtrack::core::cregion::_collect_neighbors
# :call: > stormtrack::core::cregion::_cpixel_get_neighbor
# :call: > stormtrack::core::cregion::_cpixel_unlink_region
# :call: > stormtrack::core::cregion::_cregion_extend_hole
# :call: > stormtrack::core::cregion::_cregion_extend_holes
# :call: > stormtrack::core::cregion::_cregion_extend_pixels
# :call: > stormtrack::core::cregion::_cregion_extend_pixels_nogil
# :call: > stormtrack::core::cregion::_cregion_extend_shell
# :call: > stormtrack::core::cregion::_cregion_extend_shells
# :call: > stormtrack::core::cregion::_cregion_insert_hole_pixel
# :call: > stormtrack::core::cregion::_cregion_insert_shell_pixel
# :call: > stormtrack::core::cregion::_cregion_overlap_core
# :call: > stormtrack::core::cregion::_cregion_reconnect_pixel
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_holes
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_holes_nogil
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_pixels
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_pixels_nogil
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_shells
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_shells_nogil
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::cregion::_extract_closed_path
# :call: > stormtrack::core::cregion::_find_link_to_continue
# :call: > stormtrack::core::cregion::cpixel_get_neighbor
# :call: > stormtrack::core::cregion::cpixel_set_region
# :call: > stormtrack::core::cregion::cregion_check_validity
# :call: > stormtrack::core::cregion::cregion_determine_bbox
# :call: > stormtrack::core::cregion::cregion_init
# :call: > stormtrack::core::cregion::cregion_insert_pixel
# :call: > stormtrack::core::cregion::cregion_insert_pixel_nogil
# :call: > stormtrack::core::cregion::cregion_insert_pixels_coords
# :call: > stormtrack::core::cregion::cregion_northernmost_pixel
# :call: > stormtrack::core::cregion::cregion_overlap_n_mask
# :call: > stormtrack::core::cregion::cregion_remove_pixel
# :call: > stormtrack::core::cregion::cregion_remove_pixel_nogil
# :call: > stormtrack::core::cregion::cregion_reset
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: > stormtrack::core::cregion_boundaries::categorize_boundart_is_shell
# :call: > stormtrack::core::cregion_boundaries::cpixel_angle_to_neighbor
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__init_skip
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__curr
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__prev
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle__next
# :call: > stormtrack::core::cregion_boundaries::cregions_find_northernmost_uncategorized_region
# :call: > stormtrack::core::cregions::cregions_find_connected
# :call: > stormtrack::core::grid::grid_create_pixels
# :call: > stormtrack::core::grid::grid_set_values
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::_find_background_neighbor_pixels
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds_core
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::collect_pixels
# :call: > stormtrack::core::identification::cpixel2arr
# :call: > stormtrack::core::identification::cpixel_count_neighbors_in_cregion
# :call: > stormtrack::core::identification::create_feature
# :call: > stormtrack::core::identification::cregions_merge_connected
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::grow_cregion_rec
# :call: > stormtrack::core::identification::pop_random_unassigned_pixel
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::identification::regiongrow_assign_pixel
# :call: > stormtrack::core::identification::regiongrow_resolve_multi_assignments
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: > stormtrack::core::identification::resolve_multi_assignment_best_connected_region
# :call: > stormtrack::core::identification::resolve_multi_assignment_biggest_region
# :call: > stormtrack::core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > stormtrack::core::identification::resolve_multi_assignment_strongest_region
# :call: > stormtrack::core::structs::cField2D
# :call: > stormtrack::core::structs::cGrid
# :call: > stormtrack::core::structs::cRegion
# :call: > stormtrack::core::tables::neighbor_link_stat_table_init
# :call: > stormtrack::core::tables::pixel_done_table_init
# :call: > stormtrack::core::tables::pixel_done_table_reset
# :call: > stormtrack::core::tables::pixel_region_table_init_regions
# :call: > stormtrack::core::tables::pixel_status_table_init_feature
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::pixeltype
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


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
cdef struct cField2D:
    cPixel** pixels
    np.int32_t nx
    np.int32_t ny


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::cregions_merge_connected
# :call: > stormtrack::core::identification::cregions_merge_connected_inplace
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::structs::cregion_conf_default
# :call: > stormtrack::core::cregion::cregion_init
# :call: > stormtrack::core::cregions_store::cregions_store_extend
# :call: v --- calling ---
cdef struct cRegionConf:
    int connected_max
    int pixels_max
    int shells_max
    int shell_max
    int holes_max
    int hole_max


# :call: > --- callers ---
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::cregions_store::cregions_store_extend
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegionConf
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


# :call: > --- callers ---
# :call: > stormtrack::core::cregion::_cpixel_unlink_region
# :call: > stormtrack::core::cregion::_cregion_add_connected
# :call: > stormtrack::core::cregion::_cregion_extend_hole
# :call: > stormtrack::core::cregion::_cregion_extend_holes
# :call: > stormtrack::core::cregion::_cregion_extend_pixels
# :call: > stormtrack::core::cregion::_cregion_extend_pixels_nogil
# :call: > stormtrack::core::cregion::_cregion_extend_shell
# :call: > stormtrack::core::cregion::_cregion_extend_shells
# :call: > stormtrack::core::cregion::_cregion_hole_remove_gaps
# :call: > stormtrack::core::cregion::_cregion_hole_remove_gaps_nogil
# :call: > stormtrack::core::cregion::_cregion_insert_hole_pixel
# :call: > stormtrack::core::cregion::_cregion_insert_shell_pixel
# :call: > stormtrack::core::cregion::_cregion_new_hole
# :call: > stormtrack::core::cregion::_cregion_new_shell
# :call: > stormtrack::core::cregion::_cregion_overlap_core
# :call: > stormtrack::core::cregion::_cregion_reconnect_pixel
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_holes
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_holes_nogil
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_pixels
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_pixels_nogil
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_shells
# :call: > stormtrack::core::cregion::_cregion_remove_pixel_from_shells_nogil
# :call: > stormtrack::core::cregion::_cregion_reset_connected
# :call: > stormtrack::core::cregion::_cregion_shell_remove_gaps
# :call: > stormtrack::core::cregion::_cregion_shell_remove_gaps_nogil
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::cregion::_extract_closed_path
# :call: > stormtrack::core::cregion::_find_link_to_continue
# :call: > stormtrack::core::cregion::cpixel_set_region
# :call: > stormtrack::core::cregion::cregion_check_validity
# :call: > stormtrack::core::cregion::cregion_cleanup
# :call: > stormtrack::core::cregion::cregion_connect_with
# :call: > stormtrack::core::cregion::cregion_determine_bbox
# :call: > stormtrack::core::cregion::cregion_init
# :call: > stormtrack::core::cregion::cregion_insert_pixel
# :call: > stormtrack::core::cregion::cregion_insert_pixel_nogil
# :call: > stormtrack::core::cregion::cregion_insert_pixels_coords
# :call: > stormtrack::core::cregion::cregion_merge
# :call: > stormtrack::core::cregion::cregion_northernmost_pixel
# :call: > stormtrack::core::cregion::cregion_overlap_n
# :call: > stormtrack::core::cregion::cregion_overlap_n_mask
# :call: > stormtrack::core::cregion::cregion_overlap_n_tables
# :call: > stormtrack::core::cregion::cregion_overlaps
# :call: > stormtrack::core::cregion::cregion_overlaps_tables
# :call: > stormtrack::core::cregion::cregion_pixels_remove_gaps
# :call: > stormtrack::core::cregion::cregion_pixels_remove_gaps_nogil
# :call: > stormtrack::core::cregion::cregion_remove_connected
# :call: > stormtrack::core::cregion::cregion_remove_pixel
# :call: > stormtrack::core::cregion::cregion_remove_pixel_nogil
# :call: > stormtrack::core::cregion::cregion_reset
# :call: > stormtrack::core::cregion::cregion_reset_boundaries
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: > stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: > stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle
# :call: > stormtrack::core::cregion_boundaries::cPixelCycle_init
# :call: > stormtrack::core::cregions::_cregions_check_connected
# :call: > stormtrack::core::cregions::cregions_create
# :call: > stormtrack::core::cregions::cregions_extend
# :call: > stormtrack::core::cregions::cregions_find_connected
# :call: > stormtrack::core::cregions::cregions_link_region
# :call: > stormtrack::core::cregions_store::cregions_store_extend
# :call: > stormtrack::core::cregions_store::cregions_store_get_new_region
# :call: > stormtrack::core::grid::grid_create_cregion
# :call: > stormtrack::core::identification::Feature::set_cregion
# :call: > stormtrack::core::identification::_cregion_collect_connected_regions_rec
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::_find_background_neighbor_pixels
# :call: > stormtrack::core::identification::assert_no_unambiguously_assigned_pixels
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::collect_pixels
# :call: > stormtrack::core::identification::cpixel_count_neighbors_in_cregion
# :call: > stormtrack::core::identification::cregion_collect_connected_regions
# :call: > stormtrack::core::identification::cregion_find_corresponding_feature
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::cregions_create_features
# :call: > stormtrack::core::identification::cregions_merge_connected
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::dbg_print_selected_regions
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::eliminate_regions_by_size
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_existing_region
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::initialize_surrounding_background_region
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::identification::regiongrow_assign_pixel
# :call: > stormtrack::core::identification::regiongrow_resolve_multi_assignments
# :call: > stormtrack::core::identification::resolve_multi_assignment_biggest_region
# :call: > stormtrack::core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > stormtrack::core::identification::resolve_multi_assignment_strongest_region
# :call: > stormtrack::core::structs::SuccessorCandidate
# :call: > stormtrack::core::structs::cPixel
# :call: > stormtrack::core::structs::cRegions
# :call: > stormtrack::core::structs::cRegionsStore
# :call: > stormtrack::core::tables::cregion_rank_slots_insert_region
# :call: > stormtrack::core::tables::neighbor_link_stat_table_init
# :call: > stormtrack::core::tables::neighbor_link_stat_table_reset_pixels
# :call: > stormtrack::core::tables::pixel_done_table_init
# :call: > stormtrack::core::tables::pixel_done_table_reset
# :call: > stormtrack::core::tables::pixel_region_table_alloc_pixels
# :call: > stormtrack::core::tables::pixel_region_table_cleanup_pixels
# :call: > stormtrack::core::tables::pixel_region_table_grow
# :call: > stormtrack::core::tables::pixel_region_table_init_regions
# :call: > stormtrack::core::tables::pixel_region_table_insert_region
# :call: > stormtrack::core::tables::pixel_region_table_reset_region
# :call: > stormtrack::core::tables::pixel_status_table_init_feature
# :call: > stormtrack::core::tables::pixel_status_table_reset_feature
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cPixel
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


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_cregion_collect_connected_regions_rec
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds_core
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_pixels
# :call: > stormtrack::core::identification::cregion_collect_connected_regions
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::cregions_create_features
# :call: > stormtrack::core::identification::cregions_merge_connected
# :call: > stormtrack::core::identification::cregions_merge_connected_inplace
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::eliminate_regions_by_size
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::features_neighbors_to_cregions_connected
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::grow_cregion_rec
# :call: > stormtrack::core::identification::initialize_surrounding_background_region
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::tables::pixel_region_table_init_regions
# :call: > stormtrack::core::tables::pixel_region_table_reset_regions
# :call: > stormtrack::core::tables::pixel_status_table_init_feature
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: > stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: > stormtrack::core::cregions::cregions_cleanup
# :call: > stormtrack::core::cregions::cregions_create
# :call: > stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: > stormtrack::core::cregions::cregions_extend
# :call: > stormtrack::core::cregions::cregions_find_connected
# :call: > stormtrack::core::cregion_boundaries::cregions_find_northernmost_uncategorized_region
# :call: > stormtrack::core::cregions::cregions_init
# :call: > stormtrack::core::cregions::cregions_link_region
# :call: > stormtrack::core::cregions::cregions_move
# :call: > stormtrack::core::cregions::cregions_reset
# :call: > stormtrack::core::cregions::_cregions_check_connected
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
cdef struct cRegions:
    cRegion** regions
    int n
    int max
    np.uint64_t id


# :call: > --- callers ---
# :call: > stormtrack::core::tables::neighbor_link_stat_table_init
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: v --- calling ---
@cython.cdivision
cdef inline np.uint8_t get_matching_neighbor_id(np.uint8_t ind, int nmax) nogil:
    cdef int mind = (ind + nmax / 2) % nmax
    return mind


# :call: > --- callers ---
# :call: > stormtrack::core::identification::regiongrow_resolve_multi_assignments
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: > stormtrack::core::tables::cregion_rank_slots_copy
# :call: > stormtrack::core::tables::cregion_rank_slots_extend
# :call: > stormtrack::core::tables::pixel_region_table_alloc
# :call: > stormtrack::core::tables::pixel_region_table_alloc_pixel
# :call: v --- calling ---
cdef struct cRegionRankSlot:
    cRegion* region
    np.int8_t rank


# :call: > --- callers ---
# :call: > stormtrack::core::identification::dbg_print_selected_regions
# :call: > stormtrack::core::identification::regiongrow_resolve_multi_assignments
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: > stormtrack::core::identification::resolve_multi_assignment_best_connected_region
# :call: > stormtrack::core::identification::resolve_multi_assignment_biggest_region
# :call: > stormtrack::core::identification::resolve_multi_assignment_mean_strongest_region
# :call: > stormtrack::core::identification::resolve_multi_assignment_strongest_region
# :call: > stormtrack::core::tables::cregion_rank_slots_copy
# :call: > stormtrack::core::tables::cregion_rank_slots_extend
# :call: > stormtrack::core::tables::cregion_rank_slots_insert_region
# :call: > stormtrack::core::tables::pixel_region_table_alloc
# :call: > stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: v --- calling ---
cdef struct cRegionRankSlots:
    cRegionRankSlot* slots
    int max
    int n


# :call: > --- callers ---
# :call: > stormtrack::core::tables::_pixel_region_table_cleanup_entry
# :call: > stormtrack::core::tables::pixel_region_table_alloc
# :call: > stormtrack::core::tables::pixel_region_table_alloc_grid
# :call: > stormtrack::core::tables::pixel_region_table_alloc_pixel
# :call: > stormtrack::core::tables::pixel_region_table_alloc_pixels
# :call: > stormtrack::core::tables::pixel_region_table_cleanup
# :call: > stormtrack::core::tables::pixel_region_table_cleanup_pixels
# :call: > stormtrack::core::tables::pixel_region_table_grow
# :call: > stormtrack::core::tables::pixel_region_table_init_regions
# :call: > stormtrack::core::tables::pixel_region_table_insert_region
# :call: > stormtrack::core::tables::pixel_region_table_reset
# :call: > stormtrack::core::tables::pixel_region_table_reset_region
# :call: > stormtrack::core::tables::pixel_region_table_reset_regions
# :call: > stormtrack::core::tables::pixel_region_table_reset_slots
# :call: > stormtrack::core::cregion::_cregion_overlap_core
# :call: > stormtrack::core::cregion::cregion_overlap_n_tables
# :call: > stormtrack::core::cregion::cregion_overlaps_tables
# :call: v --- calling ---
ctypedef cRegionRankSlots** PixelRegionTable


# :call: > --- callers ---
# :call: > stormtrack::core::tables::pixel_status_table_alloc
# :call: > stormtrack::core::tables::pixel_status_table_cleanup
# :call: > stormtrack::core::tables::pixel_status_table_init_feature
# :call: > stormtrack::core::tables::pixel_status_table_reset
# :call: > stormtrack::core::tables::pixel_status_table_reset_feature
# :call: v --- calling ---
ctypedef np.int8_t** PixelStatusTable


# :call: > --- callers ---
# :call: > stormtrack::core::tables::pixel_done_table_alloc
# :call: > stormtrack::core::tables::pixel_done_table_cleanup
# :call: > stormtrack::core::tables::pixel_done_table_init
# :call: > stormtrack::core::tables::pixel_done_table_reset
# :call: v --- calling ---
ctypedef bint **PixelDoneTable


# :call: > --- callers ---
# :call: > stormtrack::core::tables::neighbor_link_stat_table_alloc
# :call: > stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: > stormtrack::core::tables::neighbor_link_stat_table_reset_pixels
# :call: > stormtrack::core::tables::neighbor_link_stat_table_cleanup
# :call: > stormtrack::core::tables::neighbor_link_stat_table_init
# :call: v --- calling ---
ctypedef np.int8_t ***NeighborLinkStatTable


# :call: > --- callers ---
# :call: > stormtrack::core::structs::cGrid
# :call: > stormtrack::core::structs::cregions_store_create
# :call: > stormtrack::core::cregions_store::cregions_store_cleanup
# :call: > stormtrack::core::cregions_store::cregions_store_extend
# :call: > stormtrack::core::cregions_store::cregions_store_get_new_region
# :call: > stormtrack::core::cregions_store::cregions_store_reset
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
cdef struct cRegionsStore:
    cRegion** blocks
    int block_size
    int i_block
    int n_blocks
    int i_next_region


# :call: > --- callers ---
# :call: > stormtrack::core::structs::grid_create_empty
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegionsStore
cdef inline cRegionsStore cregions_store_create():
    return cRegionsStore(
        blocks= NULL,
        block_size= 50,
        i_block= 0,
        n_blocks= 0,
        i_next_region= 0,
        )


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds_core
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::collect_adjacent_pixels
# :call: > stormtrack::core::identification::collect_pixels
# :call: > stormtrack::core::identification::cpixel_count_neighbors_in_cregion
# :call: > stormtrack::core::identification::cregions2features_connected2neighbors
# :call: > stormtrack::core::identification::cregions_create_features
# :call: > stormtrack::core::identification::cregions_merge_connected
# :call: > stormtrack::core::identification::cregions_merge_connected_inplace
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::determine_shared_boundary_pixels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::features_find_neighbors_core
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_existing_region
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::grow_cregion_rec
# :call: > stormtrack::core::identification::init_random_seeds
# :call: > stormtrack::core::identification::initialize_surrounding_background_region
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::identification::pop_random_unassigned_pixel
# :call: > stormtrack::core::identification::regiongrow_advance_boundary
# :call: > stormtrack::core::identification::regiongrow_assign_pixel
# :call: > stormtrack::core::identification::regiongrow_resolve_multi_assignments
# :call: > stormtrack::core::identification::resolve_multi_assignment
# :call: > stormtrack::core::identification::resolve_multi_assignment_best_connected_region
# :call: > stormtrack::core::structs::grid_create_empty
# :call: > stormtrack::core::grid::Grid::to_c
# :call: > stormtrack::core::cregion_boundaries::_cregion_determine_boundaries_core
# :call: > stormtrack::core::cregion::_determine_boundary_pixels_raw
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
# :call: > stormtrack::core::cregion_boundaries::categorize_boundaries
# :call: > stormtrack::core::cregion_boundaries::cregion_determine_boundaries
# :call: > stormtrack::core::cregion_boundaries::cregions_determine_boundaries
# :call: > stormtrack::core::cregion_boundaries::categorize_boundary_is_shell
# :call: > stormtrack::core::grid::grid_cleanup
# :call: > stormtrack::core::grid::grid_create
# :call: > stormtrack::core::grid::grid_create_pixels
# :call: > stormtrack::core::grid::grid_create_cregion
# :call: > stormtrack::core::grid::grid_reset
# :call: > stormtrack::core::grid::grid_set_values
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cPixel
# :call: v stormtrack::core::structs::cRegionsStore
cdef struct cGrid:
    np.uint64_t timestep
    cConstants constants
    cPixel** pixels
    PixelRegionTable pixel_region_table
    PixelStatusTable pixel_status_table
    PixelDoneTable pixel_done_table
    NeighborLinkStatTable neighbor_link_stat_table
    cRegionsStore _regions


# :call: > --- callers ---
# :call: > stormtrack::core::grid::grid_create
# :call: > stormtrack::extra::front_surgery::*
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cConstants
# :call: v stormtrack::core::structs::cGrid
# :call: v stormtrack::core::structs::cregions_store_create
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


# :call: > --- callers ---
# :call: > stormtrack::core::structs::SuccessorCandidates
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
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


# :call: > --- callers ---
# :call: v --- calling ---
# :call: v stormtrack::core::structs::SuccessorCandidate
cdef struct SuccessorCandidates:
    SuccessorCandidate* candidates
    int n
    int max
