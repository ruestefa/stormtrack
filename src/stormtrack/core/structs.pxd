
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
# \<\(cConstants\|pixeltype\|cPixel\|cField2D\|cRegionConf\|cregion_conf_default\|cRegion\|cRegions\|get_matching_neighbor_id\|cRegionRankSlot\|cRegionRankSlots\|cRegionsStore\|cregions_store_create\|cGrid\|grid_create_empty\|SuccessorCandidate\|SuccessorCandidates\)\>


# CALL <
# CALLERS:
# > structs::cGrid
# > structs::grid_create_empty
# < identification::find_features_2d_threshold_seeded
# < identification::_find_features_threshold_random_seeds
# < identification::c_find_features_2d_threshold_seeds
# < identification::cregions_merge_connected_inplace
# < identification::cregions_merge_connected
# < identification::_cregions_merge_connected_core
# < identification::find_features_2d_threshold
# < identification::merge_adjacent_features
# < identification::feature_split_regiongrow
# < identification::features_grow
# < identification::csplit_regiongrow_levels
# < identification::features_to_cregions
# < identification::feature_to_cregion
# < identification::pixels_find_boundaries
# < identification::features_find_neighbors
# < identification::features_find_neighbors_core
# < identification::cregions_create_features
# < identification::cregions2features_connected2neighbors
# < tables::pixel_done_table_alloc
# < tables::pixel_done_table_cleanup
# < tables::pixel_region_table_alloc
# < tables::pixel_region_table_alloc_grid
# < tables::pixel_status_table_alloc
# < tables::neighbor_link_stat_table_alloc
# < tables::neighbor_link_stat_table_reset
# < tables::neighbor_link_stat_table_init
# < typedefs::Constants::__cinit__
# < typedefs::Constants::to_c
# < typedefs::_collect_neighbors
# < typedefs::cregions_find_connected
# < typedefs::grid_create
# CALLING:
# CALL >
cdef struct cConstants:
    np.int32_t nx
    np.int32_t ny
    np.uint8_t connectivity
    np.uint8_t n_neighbors_max # SR_TODO rename to pixel_neighbors_max


# CALL <
# CALLERS:
# > structs::cPixel
# < identification::grow_cregion_rec
# < typedefs::cpixels_reset
# < typedefs::cpixel2d_create
# CALLING:
# CALL >
cdef enum pixeltype:
    pixeltype_none
    pixeltype_background
    pixeltype_feature


# CALL <
# CALLERS:
# > structs::cField2D
# > structs::cRegion
# > structs::cGrid
# > identification::assign_cpixel
# > identification::cfeatures_grow_core
# > identification::c_find_features_2d_threshold_seeds_core
# > identification::collect_adjacent_pixels
# > identification::collect_pixels
# > identification::cpixel2arr
# > identification::cpixel_count_neighbors_in_cregion
# > identification::create_feature
# > identification::cregions_merge_connected
# > identification::_cregions_merge_connected_core
# > identification::csplit_regiongrow_core
# > identification::csplit_regiongrow_levels
# > identification::csplit_regiongrow_levels_core
# > identification::determine_shared_boundary_pixels
# > identification::extract_subregions_level
# > identification::feature_split_regiongrow
# > identification::_find_background_neighbor_pixels
# > identification::find_features_2d_threshold
# > identification::grow_cregion_rec
# > identification::pop_random_unassigned_pixel
# > identification::regiongrow_advance_boundary
# > identification::regiongrow_assign_pixel
# > identification::regiongrow_resolve_multi_assignments
# > identification::resolve_multi_assignment
# > identification::resolve_multi_assignment_best_connected_region
# > identification::resolve_multi_assignment_biggest_region
# > identification::resolve_multi_assignment_mean_strongest_region
# > identification::resolve_multi_assignment_strongest_region
# > tables::pixel_done_table_init
# > tables::pixel_done_table_reset
# > tables::pixel_region_table_init_regions
# > tables::pixel_status_table_init_feature
# > tables::neighbor_link_stat_table_init
# > typedefs::cregion_init
# > typedefs::cregion_insert_pixels_coords
# > typedefs::_cregion_create_pixels
# > typedefs::cregion_insert_pixel
# > typedefs::cregion_insert_pixel_nogil
# > typedefs::cregion_remove_pixel
# > typedefs::cregion_remove_pixel_nogil
# > typedefs::_cregion_remove_pixel_from_pixels
# > typedefs::_cregion_remove_pixel_from_pixels_nogil
# > typedefs::_cregion_remove_pixel_from_shells
# > typedefs::_cregion_remove_pixel_from_shells_nogil
# > typedefs::_cregion_remove_pixel_from_holes
# > typedefs::_cregion_remove_pixel_from_holes_nogil
# > typedefs::_cregion_insert_shell_pixel
# > typedefs::_cregion_insert_hole_pixel
# > typedefs::_cregion_reconnect_pixel
# > typedefs::_cregion_extend_pixels
# > typedefs::_cregion_extend_pixels_nogil
# > typedefs::_cregion_extend_shell
# > typedefs::_cregion_extend_hole
# > typedefs::_cregion_extend_shells
# > typedefs::_cregion_extend_holes
# > typedefs::cregion_reset
# > typedefs::_cpixel_unlink_region
# > typedefs::_cregion_determine_boundaries_core
# > typedefs::_reconstruct_boundaries
# > typedefs::_find_link_to_continue
# > typedefs::categorize_boundaries
# > typedefs::neighbor_pixel_angle
# > typedefs::_extract_closed_path
# > typedefs::cregion_check_validity
# > typedefs::cregion_northernmost_pixel
# > typedefs::cregions_find_northernmost_uncategorized_region
# > typedefs::_determine_boundary_pixels_raw
# > typedefs::_collect_neighbors
# > typedefs::cpixel_get_neighbor
# > typedefs::_cpixel_get_neighbor
# > typedefs::_cregion_overlap_core
# > typedefs::cregion_overlap_n_mask
# > typedefs::cregion_determine_bbox
# > typedefs::cregions_find_connected
# > typedefs::cpixel_set_region
# > typedefs::cpixels_reset
# > typedefs::cpixel2d_create
# > typedefs::grid_create_pixels
# > typedefs::grid_set_values
# CALLING:
# < structs::pixeltype
# < structs::cRegion
# CALL >
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


# CALL <
# CALLERS:
# CALLING:
# < structs::cPixel
# CALL >
cdef struct cField2D:
    cPixel** pixels
    np.int32_t nx
    np.int32_t ny


# CALL <
# CALLERS:
# > structs::cregion_conf_default
# > identification::_find_features_threshold_random_seeds
# > identification::c_find_features_2d_threshold_seeds
# > identification::cregions_merge_connected_inplace
# > identification::cregions_merge_connected
# > identification::find_features_2d_threshold
# > identification::cfeatures_grow_core
# > identification::features_to_cregions
# > identification::feature_to_cregion
# > identification::pixels_find_boundaries
# > typedefs::cregion_init
# > typedefs::cregions_store_extend
# CALLING:
# CALL >
cdef struct cRegionConf:
    int connected_max
    int pixels_max
    int shells_max
    int shell_max
    int holes_max
    int hole_max


# CALL <
# CALLERS:
# > identification::merge_adjacent_features
# > identification::feature_split_regiongrow
# > identification::features_grow
# > identification::csplit_regiongrow_levels
# > identification::extract_subregions_level
# > identification::collect_adjacent_pixels
# > identification::regiongrow_advance_boundary
# > identification::features_find_neighbors_core
# > identification::cregions2features_connected2neighbors
# > identification::determine_shared_boundary_pixels
# > typedefs::_determine_boundary_pixels_raw
# > typedefs::cregions_store_extend
# CALLING:
# < structs::cRegionConf
# CALL >
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


# CALL <
# CALLERS:
# > structs::cPixel
# > structs::cRegions
# > structs::cRegionsStore
# > structs::SuccessorCandidate
# > identification::cregions_merge_connected
# > identification::_cregions_merge_connected_core
# > identification::collect_pixels
# > identification::cregion_collect_connected_regions
# > identification::_cregion_collect_connected_regions_rec
# > identification::assign_cpixel
# > identification::find_features_2d_threshold
# > identification::eliminate_regions_by_size
# > identification::find_existing_region
# > identification::feature_split_regiongrow
# > identification::features_grow
# > identification::cfeatures_grow_core
# > identification::csplit_regiongrow_levels
# > identification::csplit_regiongrow_levels_core
# > identification::extract_subregions_level
# > identification::collect_adjacent_pixels
# > identification::csplit_regiongrow_core
# > identification::assert_no_unambiguously_assigned_pixels
# > identification::regiongrow_advance_boundary
# > identification::regiongrow_resolve_multi_assignments
# > identification::dbg_print_selected_regions
# > identification::resolve_multi_assignment_biggest_region
# > identification::resolve_multi_assignment_strongest_region
# > identification::resolve_multi_assignment_mean_strongest_region
# > identification::cpixel_count_neighbors_in_cregion
# > identification::regiongrow_assign_pixel
# > identification::features_to_cregions
# > identification::features_neighbors_to_cregions_connected
# > identification::feature_to_cregion
# > identification::pixels_find_boundaries
# > identification::Feature::set_cregion
# > identification::cregions_create_features
# > identification::cregions2features_connected2neighbors
# > identification::determine_shared_boundary_pixels
# > identification::initialize_surrounding_background_region
# > identification::_find_background_neighbor_pixels
# > identification::cregion_find_corresponding_feature
# > tables::pixel_done_table_init
# > tables::pixel_done_table_reset
# > tables::pixel_region_table_alloc_pixels
# > tables::pixel_region_table_insert_region
# > tables::cregion_rank_slots_insert_region
# > tables::pixel_region_table_init_regions
# > tables::pixel_region_table_grow
# > tables::pixel_region_table_cleanup_pixels
# > tables::pixel_region_table_reset_region
# > tables::pixel_status_table_init_feature
# > tables::pixel_status_table_reset_feature
# > tables::neighbor_link_stat_table_reset_pixels
# > tables::neighbor_link_stat_table_init
# > typedefs::cregion_init
# > typedefs::cregion_insert_pixels_coords
# > typedefs::_cregion_create_pixels
# > typedefs::cregion_insert_pixel
# > typedefs::cregion_insert_pixel_nogil
# > typedefs::cregion_remove_pixel
# > typedefs::cregion_remove_pixel_nogil
# > typedefs::_cregion_remove_pixel_from_pixels
# > typedefs::_cregion_remove_pixel_from_pixels_nogil
# > typedefs::_cregion_remove_pixel_from_shells
# > typedefs::_cregion_remove_pixel_from_shells_nogil
# > typedefs::_cregion_remove_pixel_from_holes
# > typedefs::_cregion_remove_pixel_from_holes_nogil
# > typedefs::cregion_pixels_remove_gaps
# > typedefs::cregion_pixels_remove_gaps_nogil
# > typedefs::_cregion_shell_remove_gaps
# > typedefs::_cregion_shell_remove_gaps_nogil
# > typedefs::_cregion_hole_remove_gaps
# > typedefs::_cregion_hole_remove_gaps_nogil
# > typedefs::_cregion_insert_shell_pixel
# > typedefs::_cregion_insert_hole_pixel
# > typedefs::_cregion_reconnect_pixel
# > typedefs::_cregion_extend_pixels
# > typedefs::_cregion_extend_pixels_nogil
# > typedefs::_cregion_extend_shell
# > typedefs::_cregion_extend_hole
# > typedefs::_cregion_extend_shells
# > typedefs::_cregion_extend_holes
# > typedefs::_cregion_new_shell
# > typedefs::_cregion_new_hole
# > typedefs::_cregion_add_connected
# > typedefs::cregion_reset
# > typedefs::_cpixel_unlink_region
# > typedefs::_cregion_reset_connected
# > typedefs::cregion_remove_connected
# > typedefs::cregion_cleanup
# > typedefs::cregion_merge
# > typedefs::cregion_determine_boundaries
# > typedefs::cregions_determine_boundaries
# > typedefs::_cregion_determine_boundaries_core
# > typedefs::_reconstruct_boundaries
# > typedefs::_find_link_to_continue
# > typedefs::categorize_boundaries
# > typedefs::_extract_closed_path
# > typedefs::cregion_check_validity
# > typedefs::cregion_northernmost_pixel
# > typedefs::cregion_reset_boundaries
# > typedefs::_determine_boundary_pixels_raw
# > typedefs::cregion_overlaps
# > typedefs::cregion_overlaps_tables
# > typedefs::cregion_overlap_n
# > typedefs::cregion_overlap_n_tables
# > typedefs::_cregion_overlap_core
# > typedefs::cregion_overlap_n_mask
# > typedefs::cregion_determine_bbox
# > typedefs::cregions_create
# > typedefs::cregions_link_region
# > typedefs::cregions_extend
# > typedefs::cregions_connect
# > typedefs::cregions_find_connected
# > typedefs::dbg_check_connected
# > typedefs::cpixel_set_region
# > typedefs::cregions_store_get_new_region
# > typedefs::cregions_store_extend
# > typedefs::grid_new_region
# > typedefs::grid_new_regions
# CALLING:
# < structs::cPixel
# CALL >
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


# CALL <
# CALLERS:
# > identification::_find_features_threshold_random_seeds
# > identification::c_find_features_2d_threshold_seeds
# > identification::c_find_features_2d_threshold_seeds_core
# > identification::grow_cregion_rec
# > identification::cregions_merge_connected_inplace
# > identification::cregions_merge_connected
# > identification::_cregions_merge_connected_core
# > identification::collect_pixels
# > identification::cregion_collect_connected_regions
# > identification::_cregion_collect_connected_regions_rec
# > identification::assign_cpixel
# > identification::find_features_2d_threshold
# > identification::eliminate_regions_by_size
# > identification::merge_adjacent_features
# > identification::feature_split_regiongrow
# > identification::features_grow
# > identification::cfeatures_grow_core
# > identification::csplit_regiongrow_levels
# > identification::csplit_regiongrow_levels_core
# > identification::extract_subregions_level
# > identification::csplit_regiongrow_core
# > identification::regiongrow_advance_boundary
# > identification::features_to_cregions
# > identification::features_neighbors_to_cregions_connected
# > identification::feature_to_cregion
# > identification::features_find_neighbors_core
# > identification::cregions_create_features
# > identification::cregions2features_connected2neighbors
# > identification::initialize_surrounding_background_region
# > tables::pixel_region_table_init_regions
# > tables::pixel_region_table_reset_regions
# > tables::pixel_status_table_init_feature
# > typedefs::cregion_determine_boundaries
# > typedefs::cregions_determine_boundaries
# > typedefs::_cregion_determine_boundaries_core
# > typedefs::_reconstruct_boundaries
# > typedefs::categorize_boundaries
# > typedefs::cregions_find_northernmost_uncategorized_region
# > typedefs::cregions_init
# > typedefs::cregions_create
# > typedefs::cregions_link_region
# > typedefs::cregions_extend
# > typedefs::cregions_move
# > typedefs::cregions_reset
# > typedefs::cregions_cleanup
# > typedefs::cregions_find_connected
# > typedefs::dbg_check_connected
# > typedefs::grid_new_regions
# CALLING:
# < structs::cRegion
# CALL >
cdef struct cRegions:
    cRegion** regions
    int n
    int max
    np.uint64_t id


# CALL <
# CALLERS:
# > tables::neighbor_link_stat_table_init
# > typedefs::_reconstruct_boundaries
# CALLING:
# CALL >
@cython.cdivision
cdef inline np.uint8_t get_matching_neighbor_id(np.uint8_t ind, int nmax) nogil:
    cdef int mind = (ind + nmax / 2) % nmax
    return mind


# CALL <
# CALLERS:
# > identification::regiongrow_resolve_multi_assignments
# > identification::resolve_multi_assignment
# > tables::pixel_region_table_alloc
# > tables::cregion_rank_slots_extend
# > tables::pixel_region_table_alloc_pixel
# > tables::cregion_rank_slots_copy
# CALLING:
# CALL >
cdef struct cRegionRankSlot:
    cRegion* region
    np.int8_t rank


# CALL <
# CALLERS:
# > identification::regiongrow_resolve_multi_assignments
# > identification::resolve_multi_assignment
# > identification::dbg_print_selected_regions
# > identification::resolve_multi_assignment_best_connected_region
# > identification::resolve_multi_assignment_biggest_region
# > identification::resolve_multi_assignment_strongest_region
# > identification::resolve_multi_assignment_mean_strongest_region
# > tables::pixel_region_table_alloc
# > tables::pixel_region_table_alloc_grid
# > tables::cregion_rank_slots_insert_region
# > tables::cregion_rank_slots_extend
# > tables::cregion_rank_slots_copy
# CALLING:
# CALL >
cdef struct cRegionRankSlots:
    cRegionRankSlot* slots
    int max
    int n


# CALL <
# CALLERS:
# > tables::pixel_region_table_alloc
# > tables::pixel_region_table_alloc_grid
# > tables::pixel_region_table_alloc_pixels
# > tables::pixel_region_table_insert_region
# > tables::pixel_region_table_alloc_pixel
# > tables::pixel_region_table_init_regions
# > tables::pixel_region_table_grow
# > tables::pixel_region_table_cleanup_pixels
# > tables::pixel_region_table_reset
# > tables::pixel_region_table_reset_region
# > tables::pixel_region_table_reset_regions
# > tables::pixel_region_table_reset_slots
# > tables::_pixel_region_table_cleanup_entry
# > tables::pixel_region_table_cleanup
# > typedefs::cregion_overlaps_tables
# > typedefs::cregion_overlap_n_tables
# > typedefs::_cregion_overlap_core
# CALLING:
# CALL >
ctypedef cRegionRankSlots** PixelRegionTable


# CALL <
# CALLERS:
# > tables::pixel_status_table_init_feature
# > tables::pixel_status_table_reset_feature
# > tables::pixel_status_table_alloc
# > tables::pixel_status_table_reset
# > tables::pixel_status_table_cleanup
# CALLING:
# CALL >
ctypedef np.int8_t** PixelStatusTable


# CALL <
# CALLERS:
# > tables::pixel_done_table_alloc
# > tables::pixel_done_table_init
# > tables::pixel_done_table_reset
# > tables::pixel_done_table_cleanup
# CALLING:
# CALL >
ctypedef bint **PixelDoneTable


# CALL <
# CALLERS:
# > tables::neighbor_link_stat_table_alloc
# > tables::neighbor_link_stat_table_reset
# > tables::neighbor_link_stat_table_reset_pixels
# > tables::neighbor_link_stat_table_cleanup
# > tables::neighbor_link_stat_table_init
# CALLING:
# CALL >
ctypedef np.int8_t ***NeighborLinkStatTable


# CALL <
# CALLERS:
# > structs::cregions_store_create
# > structs::cGrid
# > typedefs::cregions_store_get_new_region
# > typedefs::cregions_store_reset
# > typedefs::cregions_store_extend
# > typedefs::cregions_store_cleanup
# CALLING:
# < structs::cRegion
# CALL >
cdef struct cRegionsStore:
    cRegion** blocks
    int block_size
    int i_block
    int n_blocks
    int i_next_region


# CALL <
# CALLERS:
# > structs::grid_create_empty
# CALLING:
# < structs::cRegionsStore
# CALL >
cdef inline cRegionsStore cregions_store_create():
    return cRegionsStore(
        blocks= NULL,
        block_size= 50,
        i_block= 0,
        n_blocks= 0,
        i_next_region= 0,
        )


# CALL <
# CALLERS:
# > structs::grid_create_empty
# > identification::_find_features_threshold_random_seeds
# > identification::c_find_features_2d_threshold_seeds
# > identification::c_find_features_2d_threshold_seeds_core
# > identification::grow_cregion_rec
# > identification::init_random_seeds
# > identification::pop_random_unassigned_pixel
# > identification::cregions_merge_connected_inplace
# > identification::cregions_merge_connected
# > identification::_cregions_merge_connected_core
# > identification::collect_pixels
# > identification::assign_cpixel
# > identification::find_features_2d_threshold
# > identification::find_existing_region
# > identification::merge_adjacent_features
# > identification::feature_split_regiongrow
# > identification::features_grow
# > identification::cfeatures_grow_core
# > identification::csplit_regiongrow_levels
# > identification::csplit_regiongrow_levels_core
# > identification::extract_subregions_level
# > identification::collect_adjacent_pixels
# > identification::csplit_regiongrow_core
# > identification::regiongrow_advance_boundary
# > identification::regiongrow_resolve_multi_assignments
# > identification::resolve_multi_assignment
# > identification::resolve_multi_assignment_best_connected_region
# > identification::cpixel_count_neighbors_in_cregion
# > identification::regiongrow_assign_pixel
# > identification::features_to_cregions
# > identification::feature_to_cregion
# > identification::pixels_find_boundaries
# > identification::features_find_neighbors
# > identification::features_find_neighbors_core
# > identification::cregions_create_features
# > identification::cregions2features_connected2neighbors
# > identification::determine_shared_boundary_pixels
# > identification::initialize_surrounding_background_region
# > typedefs::cregion_determine_boundaries
# > typedefs::cregions_determine_boundaries
# > typedefs::_cregion_determine_boundaries_core
# > typedefs::_reconstruct_boundaries
# > typedefs::categorize_boundaries
# > typedefs::boundary_must_be_a_shell
# > typedefs::_determine_boundary_pixels_raw
# > typedefs::Grid::to_c
# > typedefs::grid_create
# > typedefs::grid_reset
# > typedefs::grid_cleanup
# > typedefs::grid_create_pixels
# > typedefs::grid_set_values
# > typedefs::grid_new_region
# > typedefs::grid_new_regions
# CALLING:
# < structs::cConstants
# < structs::cPixel
# < structs::cRegionsStore
# CALL >
cdef struct cGrid:
    np.uint64_t timestep
    cConstants constants
    cPixel** pixels
    PixelRegionTable pixel_region_table
    PixelStatusTable pixel_status_table
    PixelDoneTable pixel_done_table
    NeighborLinkStatTable neighbor_link_stat_table
    cRegionsStore _regions


# CALL <
# CALLERS:
# > typedefs::grid_create
# CALLING:
# < structs::cGrid
# < structs::cConstants
# < structs::cregions_store_create
# CALL >
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


# CALL <
# CALLERS:
# > structs::SuccessorCandidates
# CALLING:
# < structs::cRegion
# CALL >
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


# CALL <
# CALLERS:
# CALLING:
# < structs::SuccessorCandidate
# CALL >
cdef struct SuccessorCandidates:
    SuccessorCandidate* candidates
    int n
    int max
