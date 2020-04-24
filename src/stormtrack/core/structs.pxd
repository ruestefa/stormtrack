
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
# :call: > structs::cGrid
# :call: > structs::grid_create_empty
# :call: v --- CALLING ---
# :call: v identification::_cregions_merge_connected_core
# :call: v identification::_find_features_threshold_random_seeds
# :call: v identification::c_find_features_2d_threshold_seeds
# :call: v identification::cregions2features_connected2neighbors
# :call: v identification::cregions_create_features
# :call: v identification::cregions_merge_connected
# :call: v identification::cregions_merge_connected_inplace
# :call: v identification::csplit_regiongrow_levels
# :call: v identification::feature_split_regiongrow
# :call: v identification::feature_to_cregion
# :call: v identification::features_find_neighbors
# :call: v identification::features_find_neighbors_core
# :call: v identification::features_grow
# :call: v identification::features_to_cregions
# :call: v identification::find_features_2d_threshold
# :call: v identification::find_features_2d_threshold_seeded
# :call: v identification::merge_adjacent_features
# :call: v identification::pixels_find_boundaries
# :call: v tables::neighbor_link_stat_table_alloc
# :call: v tables::neighbor_link_stat_table_init
# :call: v tables::neighbor_link_stat_table_reset
# :call: v tables::pixel_done_table_alloc
# :call: v tables::pixel_done_table_cleanup
# :call: v tables::pixel_region_table_alloc
# :call: v tables::pixel_region_table_alloc_grid
# :call: v tables::pixel_status_table_alloc
# :call: v typedefs::Constants::__cinit__
# :call: v typedefs::Constants::to_c
# :call: v typedefs::_collect_neighbors
# :call: v typedefs::cregions_find_connected
# :call: v typedefs::grid_create
cdef struct cConstants:
    np.int32_t nx
    np.int32_t ny
    np.uint8_t connectivity
    np.uint8_t n_neighbors_max # SR_TODO rename to pixel_neighbors_max


# :call: > --- CALLERS ---
# :call: > structs::cPixel
# :call: v --- CALLING ---
# :call: v identification::grow_cregion_rec
# :call: v typedefs::cpixel2d_create
# :call: v typedefs::cpixels_reset
cdef enum pixeltype:
    pixeltype_none
    pixeltype_background
    pixeltype_feature


# :call: > --- CALLERS ---
# :call: > identification::_cregions_merge_connected_core
# :call: > identification::_find_background_neighbor_pixels
# :call: > identification::assign_cpixel
# :call: > identification::c_find_features_2d_threshold_seeds_core
# :call: > identification::cfeatures_grow_core
# :call: > identification::collect_adjacent_pixels
# :call: > identification::collect_pixels
# :call: > identification::cpixel2arr
# :call: > identification::cpixel_count_neighbors_in_cregion
# :call: > identification::create_feature
# :call: > identification::cregions_merge_connected
# :call: > identification::csplit_regiongrow_core
# :call: > identification::csplit_regiongrow_levels
# :call: > identification::csplit_regiongrow_levels_core
# :call: > identification::determine_shared_boundary_pixels
# :call: > identification::extract_subregions_level
# :call: > identification::feature_split_regiongrow
# :call: > identification::find_features_2d_threshold
# :call: > identification::grow_cregion_rec
# :call: > identification::pop_random_unassigned_pixel
# :call: > identification::regiongrow_advance_boundary
# :call: > identification::regiongrow_assign_pixel
# :call: > identification::regiongrow_resolve_multi_assignments
# :call: > identification::resolve_multi_assignment
# :call: > identification::resolve_multi_assignment_best_connected_region
# :call: > identification::resolve_multi_assignment_biggest_region
# :call: > identification::resolve_multi_assignment_mean_strongest_region
# :call: > identification::resolve_multi_assignment_strongest_region
# :call: > structs::cField2D
# :call: > structs::cGrid
# :call: > structs::cRegion
# :call: > tables::neighbor_link_stat_table_init
# :call: > tables::pixel_done_table_init
# :call: > tables::pixel_done_table_reset
# :call: > tables::pixel_region_table_init_regions
# :call: > tables::pixel_status_table_init_feature
# :call: > typedefs::_collect_neighbors
# :call: > typedefs::_cpixel_get_neighbor
# :call: > typedefs::_cpixel_unlink_region
# :call: > typedefs::_cregion_create_pixels
# :call: > typedefs::_cregion_determine_boundaries_core
# :call: > typedefs::_cregion_extend_hole
# :call: > typedefs::_cregion_extend_holes
# :call: > typedefs::_cregion_extend_pixels
# :call: > typedefs::_cregion_extend_pixels_nogil
# :call: > typedefs::_cregion_extend_shell
# :call: > typedefs::_cregion_extend_shells
# :call: > typedefs::_cregion_insert_hole_pixel
# :call: > typedefs::_cregion_insert_shell_pixel
# :call: > typedefs::_cregion_overlap_core
# :call: > typedefs::_cregion_reconnect_pixel
# :call: > typedefs::_cregion_remove_pixel_from_holes
# :call: > typedefs::_cregion_remove_pixel_from_holes_nogil
# :call: > typedefs::_cregion_remove_pixel_from_pixels
# :call: > typedefs::_cregion_remove_pixel_from_pixels_nogil
# :call: > typedefs::_cregion_remove_pixel_from_shells
# :call: > typedefs::_cregion_remove_pixel_from_shells_nogil
# :call: > typedefs::_determine_boundary_pixels_raw
# :call: > typedefs::_extract_closed_path
# :call: > typedefs::_find_link_to_continue
# :call: > typedefs::_reconstruct_boundaries
# :call: > typedefs::categorize_boundaries
# :call: > typedefs::cpixel2d_create
# :call: > typedefs::cpixel_get_neighbor
# :call: > typedefs::cpixel_set_region
# :call: > typedefs::cpixels_reset
# :call: > typedefs::cregion_check_validity
# :call: > typedefs::cregion_determine_bbox
# :call: > typedefs::cregion_init
# :call: > typedefs::cregion_insert_pixel
# :call: > typedefs::cregion_insert_pixel_nogil
# :call: > typedefs::cregion_insert_pixels_coords
# :call: > typedefs::cregion_northernmost_pixel
# :call: > typedefs::cregion_overlap_n_mask
# :call: > typedefs::cregion_remove_pixel
# :call: > typedefs::cregion_remove_pixel_nogil
# :call: > typedefs::cregion_reset
# :call: > typedefs::cregions_find_connected
# :call: > typedefs::cregions_find_northernmost_uncategorized_region
# :call: > typedefs::grid_create_pixels
# :call: > typedefs::grid_set_values
# :call: > typedefs::neighbor_pixel_angle
# :call: v --- CALLING ---
# :call: v structs::cRegion
# :call: v structs::pixeltype
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
# :call: v structs::cPixel
cdef struct cField2D:
    cPixel** pixels
    np.int32_t nx
    np.int32_t ny


# :call: > --- CALLERS ---
# :call: > identification::_find_features_threshold_random_seeds
# :call: > identification::c_find_features_2d_threshold_seeds
# :call: > identification::cfeatures_grow_core
# :call: > identification::cregions_merge_connected
# :call: > identification::cregions_merge_connected_inplace
# :call: > identification::feature_to_cregion
# :call: > identification::features_to_cregions
# :call: > identification::find_features_2d_threshold
# :call: > identification::pixels_find_boundaries
# :call: > structs::cregion_conf_default
# :call: > typedefs::cregion_init
# :call: > typedefs::cregions_store_extend
# :call: v --- CALLING ---
cdef struct cRegionConf:
    int connected_max
    int pixels_max
    int shells_max
    int shell_max
    int holes_max
    int hole_max


# :call: > --- CALLERS ---
# :call: > identification::collect_adjacent_pixels
# :call: > identification::cregions2features_connected2neighbors
# :call: > identification::csplit_regiongrow_levels
# :call: > identification::determine_shared_boundary_pixels
# :call: > identification::extract_subregions_level
# :call: > identification::feature_split_regiongrow
# :call: > identification::features_find_neighbors_core
# :call: > identification::features_grow
# :call: > identification::merge_adjacent_features
# :call: > identification::regiongrow_advance_boundary
# :call: > typedefs::_determine_boundary_pixels_raw
# :call: > typedefs::cregions_store_extend
# :call: v --- CALLING ---
# :call: v structs::cRegionConf
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
# :call: > identification::Feature::set_cregion
# :call: > identification::_cregion_collect_connected_regions_rec
# :call: > identification::_cregions_merge_connected_core
# :call: > identification::_find_background_neighbor_pixels
# :call: > identification::assert_no_unambiguously_assigned_pixels
# :call: > identification::assign_cpixel
# :call: > identification::cfeatures_grow_core
# :call: > identification::collect_adjacent_pixels
# :call: > identification::collect_pixels
# :call: > identification::cpixel_count_neighbors_in_cregion
# :call: > identification::cregion_collect_connected_regions
# :call: > identification::cregion_find_corresponding_feature
# :call: > identification::cregions2features_connected2neighbors
# :call: > identification::cregions_create_features
# :call: > identification::cregions_merge_connected
# :call: > identification::csplit_regiongrow_core
# :call: > identification::csplit_regiongrow_levels
# :call: > identification::csplit_regiongrow_levels_core
# :call: > identification::dbg_print_selected_regions
# :call: > identification::determine_shared_boundary_pixels
# :call: > identification::eliminate_regions_by_size
# :call: > identification::extract_subregions_level
# :call: > identification::feature_split_regiongrow
# :call: > identification::feature_to_cregion
# :call: > identification::features_grow
# :call: > identification::features_neighbors_to_cregions_connected
# :call: > identification::features_to_cregions
# :call: > identification::find_existing_region
# :call: > identification::find_features_2d_threshold
# :call: > identification::initialize_surrounding_background_region
# :call: > identification::pixels_find_boundaries
# :call: > identification::regiongrow_advance_boundary
# :call: > identification::regiongrow_assign_pixel
# :call: > identification::regiongrow_resolve_multi_assignments
# :call: > identification::resolve_multi_assignment_biggest_region
# :call: > identification::resolve_multi_assignment_mean_strongest_region
# :call: > identification::resolve_multi_assignment_strongest_region
# :call: > structs::SuccessorCandidate
# :call: > structs::cPixel
# :call: > structs::cRegions
# :call: > structs::cRegionsStore
# :call: > tables::cregion_rank_slots_insert_region
# :call: > tables::neighbor_link_stat_table_init
# :call: > tables::neighbor_link_stat_table_reset_pixels
# :call: > tables::pixel_done_table_init
# :call: > tables::pixel_done_table_reset
# :call: > tables::pixel_region_table_alloc_pixels
# :call: > tables::pixel_region_table_cleanup_pixels
# :call: > tables::pixel_region_table_grow
# :call: > tables::pixel_region_table_init_regions
# :call: > tables::pixel_region_table_insert_region
# :call: > tables::pixel_region_table_reset_region
# :call: > tables::pixel_status_table_init_feature
# :call: > tables::pixel_status_table_reset_feature
# :call: > typedefs::_cpixel_unlink_region
# :call: > typedefs::_cregion_add_connected
# :call: > typedefs::_cregion_create_pixels
# :call: > typedefs::_cregion_determine_boundaries_core
# :call: > typedefs::_cregion_extend_hole
# :call: > typedefs::_cregion_extend_holes
# :call: > typedefs::_cregion_extend_pixels
# :call: > typedefs::_cregion_extend_pixels_nogil
# :call: > typedefs::_cregion_extend_shell
# :call: > typedefs::_cregion_extend_shells
# :call: > typedefs::_cregion_hole_remove_gaps
# :call: > typedefs::_cregion_hole_remove_gaps_nogil
# :call: > typedefs::_cregion_insert_hole_pixel
# :call: > typedefs::_cregion_insert_shell_pixel
# :call: > typedefs::_cregion_new_hole
# :call: > typedefs::_cregion_new_shell
# :call: > typedefs::_cregion_overlap_core
# :call: > typedefs::_cregion_reconnect_pixel
# :call: > typedefs::_cregion_remove_pixel_from_holes
# :call: > typedefs::_cregion_remove_pixel_from_holes_nogil
# :call: > typedefs::_cregion_remove_pixel_from_pixels
# :call: > typedefs::_cregion_remove_pixel_from_pixels_nogil
# :call: > typedefs::_cregion_remove_pixel_from_shells
# :call: > typedefs::_cregion_remove_pixel_from_shells_nogil
# :call: > typedefs::_cregion_reset_connected
# :call: > typedefs::_cregion_shell_remove_gaps
# :call: > typedefs::_cregion_shell_remove_gaps_nogil
# :call: > typedefs::_determine_boundary_pixels_raw
# :call: > typedefs::_extract_closed_path
# :call: > typedefs::_find_link_to_continue
# :call: > typedefs::_reconstruct_boundaries
# :call: > typedefs::categorize_boundaries
# :call: > typedefs::cpixel_set_region
# :call: > typedefs::cregion_check_validity
# :call: > typedefs::cregion_cleanup
# :call: > typedefs::cregion_determine_bbox
# :call: > typedefs::cregion_determine_boundaries
# :call: > typedefs::cregion_init
# :call: > typedefs::cregion_insert_pixel
# :call: > typedefs::cregion_insert_pixel_nogil
# :call: > typedefs::cregion_insert_pixels_coords
# :call: > typedefs::cregion_merge
# :call: > typedefs::cregion_northernmost_pixel
# :call: > typedefs::cregion_overlap_n
# :call: > typedefs::cregion_overlap_n_mask
# :call: > typedefs::cregion_overlap_n_tables
# :call: > typedefs::cregion_overlaps
# :call: > typedefs::cregion_overlaps_tables
# :call: > typedefs::cregion_pixels_remove_gaps
# :call: > typedefs::cregion_pixels_remove_gaps_nogil
# :call: > typedefs::cregion_remove_connected
# :call: > typedefs::cregion_remove_pixel
# :call: > typedefs::cregion_remove_pixel_nogil
# :call: > typedefs::cregion_reset
# :call: > typedefs::cregion_reset_boundaries
# :call: > typedefs::cregions_connect
# :call: > typedefs::cregions_create
# :call: > typedefs::cregions_determine_boundaries
# :call: > typedefs::cregions_extend
# :call: > typedefs::cregions_find_connected
# :call: > typedefs::cregions_link_region
# :call: > typedefs::cregions_store_extend
# :call: > typedefs::cregions_store_get_new_region
# :call: > typedefs::dbg_check_connected
# :call: > typedefs::grid_new_region
# :call: > typedefs::grid_new_regions
# :call: v --- CALLING ---
# :call: v structs::cPixel
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
# :call: > identification::_cregion_collect_connected_regions_rec
# :call: > identification::_cregions_merge_connected_core
# :call: > identification::_find_features_threshold_random_seeds
# :call: > identification::assign_cpixel
# :call: > identification::c_find_features_2d_threshold_seeds
# :call: > identification::c_find_features_2d_threshold_seeds_core
# :call: > identification::cfeatures_grow_core
# :call: > identification::collect_pixels
# :call: > identification::cregion_collect_connected_regions
# :call: > identification::cregions2features_connected2neighbors
# :call: > identification::cregions_create_features
# :call: > identification::cregions_merge_connected
# :call: > identification::cregions_merge_connected_inplace
# :call: > identification::csplit_regiongrow_core
# :call: > identification::csplit_regiongrow_levels
# :call: > identification::csplit_regiongrow_levels_core
# :call: > identification::eliminate_regions_by_size
# :call: > identification::extract_subregions_level
# :call: > identification::feature_split_regiongrow
# :call: > identification::feature_to_cregion
# :call: > identification::features_find_neighbors_core
# :call: > identification::features_grow
# :call: > identification::features_neighbors_to_cregions_connected
# :call: > identification::features_to_cregions
# :call: > identification::find_features_2d_threshold
# :call: > identification::grow_cregion_rec
# :call: > identification::initialize_surrounding_background_region
# :call: > identification::merge_adjacent_features
# :call: > identification::regiongrow_advance_boundary
# :call: > tables::pixel_region_table_init_regions
# :call: > tables::pixel_region_table_reset_regions
# :call: > tables::pixel_status_table_init_feature
# :call: > typedefs::_cregion_determine_boundaries_core
# :call: > typedefs::_reconstruct_boundaries
# :call: > typedefs::categorize_boundaries
# :call: > typedefs::cregion_determine_boundaries
# :call: > typedefs::cregions_cleanup
# :call: > typedefs::cregions_create
# :call: > typedefs::cregions_determine_boundaries
# :call: > typedefs::cregions_extend
# :call: > typedefs::cregions_find_connected
# :call: > typedefs::cregions_find_northernmost_uncategorized_region
# :call: > typedefs::cregions_init
# :call: > typedefs::cregions_link_region
# :call: > typedefs::cregions_move
# :call: > typedefs::cregions_reset
# :call: > typedefs::dbg_check_connected
# :call: > typedefs::grid_new_regions
# :call: v --- CALLING ---
# :call: v structs::cRegion
cdef struct cRegions:
    cRegion** regions
    int n
    int max
    np.uint64_t id


# :call: > --- CALLERS ---
# :call: > tables::neighbor_link_stat_table_init
# :call: > typedefs::_reconstruct_boundaries
# :call: v --- CALLING ---
@cython.cdivision
cdef inline np.uint8_t get_matching_neighbor_id(np.uint8_t ind, int nmax) nogil:
    cdef int mind = (ind + nmax / 2) % nmax
    return mind


# :call: > --- CALLERS ---
# :call: > identification::regiongrow_resolve_multi_assignments
# :call: > identification::resolve_multi_assignment
# :call: > tables::cregion_rank_slots_copy
# :call: > tables::cregion_rank_slots_extend
# :call: > tables::pixel_region_table_alloc
# :call: > tables::pixel_region_table_alloc_pixel
# :call: v --- CALLING ---
cdef struct cRegionRankSlot:
    cRegion* region
    np.int8_t rank


# :call: > --- CALLERS ---
# :call: > identification::dbg_print_selected_regions
# :call: > identification::regiongrow_resolve_multi_assignments
# :call: > identification::resolve_multi_assignment
# :call: > identification::resolve_multi_assignment_best_connected_region
# :call: > identification::resolve_multi_assignment_biggest_region
# :call: > identification::resolve_multi_assignment_mean_strongest_region
# :call: > identification::resolve_multi_assignment_strongest_region
# :call: > tables::cregion_rank_slots_copy
# :call: > tables::cregion_rank_slots_extend
# :call: > tables::cregion_rank_slots_insert_region
# :call: > tables::pixel_region_table_alloc
# :call: > tables::pixel_region_table_alloc_grid
# :call: v --- CALLING ---
cdef struct cRegionRankSlots:
    cRegionRankSlot* slots
    int max
    int n


# :call: > --- CALLERS ---
# :call: > tables::_pixel_region_table_cleanup_entry
# :call: > tables::pixel_region_table_alloc
# :call: > tables::pixel_region_table_alloc_grid
# :call: > tables::pixel_region_table_alloc_pixel
# :call: > tables::pixel_region_table_alloc_pixels
# :call: > tables::pixel_region_table_cleanup
# :call: > tables::pixel_region_table_cleanup_pixels
# :call: > tables::pixel_region_table_grow
# :call: > tables::pixel_region_table_init_regions
# :call: > tables::pixel_region_table_insert_region
# :call: > tables::pixel_region_table_reset
# :call: > tables::pixel_region_table_reset_region
# :call: > tables::pixel_region_table_reset_regions
# :call: > tables::pixel_region_table_reset_slots
# :call: > typedefs::_cregion_overlap_core
# :call: > typedefs::cregion_overlap_n_tables
# :call: > typedefs::cregion_overlaps_tables
# :call: v --- CALLING ---
ctypedef cRegionRankSlots** PixelRegionTable


# :call: > --- CALLERS ---
# :call: > tables::pixel_status_table_alloc
# :call: > tables::pixel_status_table_cleanup
# :call: > tables::pixel_status_table_init_feature
# :call: > tables::pixel_status_table_reset
# :call: > tables::pixel_status_table_reset_feature
# :call: v --- CALLING ---
ctypedef np.int8_t** PixelStatusTable


# :call: > --- CALLERS ---
# :call: > tables::pixel_done_table_alloc
# :call: > tables::pixel_done_table_cleanup
# :call: > tables::pixel_done_table_init
# :call: > tables::pixel_done_table_reset
# :call: v --- CALLING ---
ctypedef bint **PixelDoneTable


# :call: > --- CALLERS ---
# :call: > tables::neighbor_link_stat_table_alloc
# :call: > tables::neighbor_link_stat_table_reset
# :call: > tables::neighbor_link_stat_table_reset_pixels
# :call: > tables::neighbor_link_stat_table_cleanup
# :call: > tables::neighbor_link_stat_table_init
# :call: v --- CALLING ---
ctypedef np.int8_t ***NeighborLinkStatTable


# :call: > --- CALLERS ---
# :call: > structs::cGrid
# :call: > structs::cregions_store_create
# :call: > typedefs::cregions_store_cleanup
# :call: > typedefs::cregions_store_extend
# :call: > typedefs::cregions_store_get_new_region
# :call: > typedefs::cregions_store_reset
# :call: v --- CALLING ---
# :call: v structs::cRegion
cdef struct cRegionsStore:
    cRegion** blocks
    int block_size
    int i_block
    int n_blocks
    int i_next_region


# :call: > --- CALLERS ---
# :call: > structs::grid_create_empty
# :call: v --- CALLING ---
# :call: v structs::cRegionsStore
cdef inline cRegionsStore cregions_store_create():
    return cRegionsStore(
        blocks= NULL,
        block_size= 50,
        i_block= 0,
        n_blocks= 0,
        i_next_region= 0,
        )


# :call: > --- CALLERS ---
# :call: > identification::_cregions_merge_connected_core
# :call: > identification::_find_features_threshold_random_seeds
# :call: > identification::assign_cpixel
# :call: > identification::c_find_features_2d_threshold_seeds
# :call: > identification::c_find_features_2d_threshold_seeds_core
# :call: > identification::cfeatures_grow_core
# :call: > identification::collect_adjacent_pixels
# :call: > identification::collect_pixels
# :call: > identification::cpixel_count_neighbors_in_cregion
# :call: > identification::cregions2features_connected2neighbors
# :call: > identification::cregions_create_features
# :call: > identification::cregions_merge_connected
# :call: > identification::cregions_merge_connected_inplace
# :call: > identification::csplit_regiongrow_core
# :call: > identification::csplit_regiongrow_levels
# :call: > identification::csplit_regiongrow_levels_core
# :call: > identification::determine_shared_boundary_pixels
# :call: > identification::extract_subregions_level
# :call: > identification::feature_split_regiongrow
# :call: > identification::feature_to_cregion
# :call: > identification::features_find_neighbors
# :call: > identification::features_find_neighbors_core
# :call: > identification::features_grow
# :call: > identification::features_to_cregions
# :call: > identification::find_existing_region
# :call: > identification::find_features_2d_threshold
# :call: > identification::grow_cregion_rec
# :call: > identification::init_random_seeds
# :call: > identification::initialize_surrounding_background_region
# :call: > identification::merge_adjacent_features
# :call: > identification::pixels_find_boundaries
# :call: > identification::pop_random_unassigned_pixel
# :call: > identification::regiongrow_advance_boundary
# :call: > identification::regiongrow_assign_pixel
# :call: > identification::regiongrow_resolve_multi_assignments
# :call: > identification::resolve_multi_assignment
# :call: > identification::resolve_multi_assignment_best_connected_region
# :call: > structs::grid_create_empty
# :call: > typedefs::Grid::to_c
# :call: > typedefs::_cregion_determine_boundaries_core
# :call: > typedefs::_determine_boundary_pixels_raw
# :call: > typedefs::_reconstruct_boundaries
# :call: > typedefs::boundary_must_be_a_shell
# :call: > typedefs::categorize_boundaries
# :call: > typedefs::cregion_determine_boundaries
# :call: > typedefs::cregions_determine_boundaries
# :call: > typedefs::grid_cleanup
# :call: > typedefs::grid_create
# :call: > typedefs::grid_create_pixels
# :call: > typedefs::grid_new_region
# :call: > typedefs::grid_new_regions
# :call: > typedefs::grid_reset
# :call: > typedefs::grid_set_values
# :call: v --- CALLING ---
# :call: v structs::cConstants
# :call: v structs::cPixel
# :call: v structs::cRegionsStore
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
# :call: > typedefs::grid_create
# :call: v --- CALLING ---
# :call: v structs::cConstants
# :call: v structs::cGrid
# :call: v structs::cregions_store_create
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
# :call: > structs::SuccessorCandidates
# :call: v --- CALLING ---
# :call: v structs::cRegion
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
# :call: v structs::SuccessorCandidate
cdef struct SuccessorCandidates:
    SuccessorCandidate* candidates
    int n
    int max
