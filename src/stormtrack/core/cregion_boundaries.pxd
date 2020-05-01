# Third-party
cimport numpy as np

# Local
from .structs cimport cGrid
from .structs cimport cPixel
from .structs cimport cRegion
from .structs cimport cRegions
from .structs cimport get_matching_neighbor_id


# SR_TMP <
from .cregion cimport _cregion_insert_hole_pixel
from .cregion cimport _cregion_insert_shell_pixel
from .cregion cimport _cregion_new_hole
from .cregion cimport _cregion_new_shell
from .cregion cimport _determine_boundary_pixels_raw
from .cregion cimport _extract_closed_path
from .cregion cimport _find_link_to_continue
from .cregion cimport cpixel_set_region
from .cregion cimport cregion_cleanup
from .cregion cimport cregion_insert_pixel
from .cregion cimport cregion_northernmost_pixel
from .cregion cimport cregion_reset_boundaries
from .cregions cimport categorize_boundaries
from .cregions cimport cregions_cleanup
from .cregions cimport cregions_create
from .cregions cimport cregions_link_region
from .grid cimport grid_create_cregion
from .tables cimport neighbor_link_stat_table_init
from .tables cimport neighbor_link_stat_table_reset
# SR_TMP >


# :call: > --- callers ---
# :call: > stormtrack::core::identification::_cregions_merge_connected_core
# :call: > stormtrack::core::identification::feature_to_cregion
# :call: > stormtrack::core::identification::pixels_find_boundaries
# :call: > stormtrack::extra::front_surgery::*
cdef void cregion_determine_boundaries(cRegion* cregion, cGrid* grid) except *


# :call: > --- callers ---
# :call: > stormtrack::core::identification::cfeatures_grow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_core
# :call: > stormtrack::core::identification::csplit_regiongrow_levels
# :call: > stormtrack::core::identification::extract_subregions_level
# :call: > stormtrack::core::identification::features_to_cregions
# :call: > stormtrack::core::identification::find_features_2d_threshold
# :call: > stormtrack::core::cregion_boundaries::cregion_determine_boundaries
cdef void cregions_determine_boundaries(cRegions* cregions, cGrid* grid) except *
