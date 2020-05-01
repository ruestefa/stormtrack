
# Third-party
cimport numpy as np

# Local
from .structs cimport cConstants
from .structs cimport cGrid
from .structs cimport cPixel
from .structs cimport cRegion
from .structs cimport grid_create_empty

# SR_TMP <
from .constants cimport Constants
from .cpixel cimport cpixel2d_create
from .cpixel cimport cpixels_reset
from .cregion cimport _collect_neighbors
from .cregions_store cimport cregions_store_cleanup
from .cregions_store cimport cregions_store_get_new_region
from .cregions_store cimport cregions_store_reset
from .tables cimport neighbor_link_stat_table_alloc
from .tables cimport neighbor_link_stat_table_cleanup
from .tables cimport neighbor_link_stat_table_reset
from .tables cimport pixel_done_table_cleanup
from .tables cimport pixel_region_table_alloc
from .tables cimport pixel_region_table_cleanup
from .tables cimport pixel_region_table_reset
from .tables cimport pixel_status_table_alloc
from .tables cimport pixel_status_table_cleanup
from .tables cimport pixel_status_table_reset
# SR_TMP >

# SR_TMP <
cdef void grid_set_values(cGrid* grid, np.float32_t[:, :] fld) except *
# SR_TMP >


# :call: > --- callers ---
# :call: > stormtrack::core::identification::features_grow
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::identification::identify_features
# :call: > stormtrack::core::tracking::FeatureTracker::__cinit__
# :call: > stormtrack::core::tracking::FeatureTracker::_swap_grids
# :call: > stormtrack::track_features::*
cdef class Grid:
    cdef readonly:
        Constants constants

    cdef:
        cGrid _cgrid

    cdef cGrid* to_c(self)

    cpdef void set_values(self, np.ndarray[np.float32_t, ndim=2] fld)

    cpdef void reset(self)

    cdef void reset_tables(self)


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::grid::Grid::__cinit__
# :call: > stormtrack::extra::front_surgery::*
cdef cGrid grid_create(np.float32_t[:, :] fld, cConstants constants) except *


# :call: > --- callers ---
# :call: > stormtrack::core::grid::grid_create
# :call: > stormtrack::extra::front_surgery::*
cdef void grid_create_pixels(cGrid* grid, np.float32_t[:, :] fld) except *


# :call: > --- CALLING ---
# :call: > stormtrack::core::structs::cGrid
# :call: > stormtrack::core::tables::neighbor_link_stat_table_reset
# :call: > stormtrack::core::tables::pixel_region_table_reset
# :call: > stormtrack::core::tables::pixel_status_table_reset
# :call: > stormtrack::core::cpixel::cpixels_reset
# :call: > stormtrack::core::cregions_store::cregions_store_reset
# :call: > stormtrack::extra::front_surgery::*
cdef void grid_reset(cGrid* grid) except *


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_find_features_threshold_random_seeds
# :call: > stormtrack::core::identification::c_find_features_2d_threshold_seeds
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::feature_split_regiongrow
# :call: > stormtrack::core::identification::features_find_neighbors
# :call: > stormtrack::core::identification::merge_adjacent_features
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::core::grid::Grid::__dealloc__
# :call: > stormtrack::extra::front_surgery::*
cdef void grid_cleanup(cGrid* grid) except *


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::assign_cpixel
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::csplit_regiongrow_levels_core
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::cregion_boundaries::_reconstruct_boundaries
cdef cRegion* grid_create_cregion(cGrid* grid) except *
