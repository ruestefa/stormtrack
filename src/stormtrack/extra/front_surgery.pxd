
cimport numpy as np

from ..core.identification cimport Feature
from ..core.identification cimport cregions_create_features
from ..core.identification cimport cregions_merge_connected_inplace
from ..core.identification cimport features_reset_cregion
from ..core.identification cimport features_to_cregions
from ..core.identification cimport features_find_neighbors
from ..core.structs cimport cConstants
from ..core.typedefs cimport Constants
from ..core.structs cimport cGrid
from ..core.tables cimport neighbor_link_stat_table_alloc
from ..core.tables cimport pixel_region_table_alloc_grid
from ..core.tables cimport pixel_status_table_alloc
from ..core.typedefs cimport cRegion
from ..core.typedefs cimport cRegions
from ..core.typedefs cimport cregion_conf_default
from ..core.typedefs cimport cregion_determine_boundaries
from ..core.typedefs cimport cregion_overlap_n
from ..core.typedefs cimport cregions_cleanup
from ..core.typedefs cimport cregions_create
from ..core.typedefs cimport cregions_find_connected
from ..core.typedefs cimport cregions_link_region
from ..core.typedefs cimport grid_cleanup
from ..core.typedefs cimport grid_create
from ..core.typedefs cimport grid_create_empty
from ..core.typedefs cimport grid_create_pixels
from ..core.typedefs cimport grid_reset


ctypedef enum category:
    category_clutter0
    category_clutter1
    category_small
    category_medium
    category_large
