
# Third-party
# cimport numpy as np

# Local
from .structs cimport cGrid
from .structs cimport cRegion
from .structs cimport cRegionConf
from .structs cimport cRegionsStore
from .structs cimport cregion_conf_default


# SR_TMP <
from .cregion cimport cregion_cleanup
from .cregion cimport cregion_get_unique_id
from .cregion cimport cregion_init
from .cregion cimport cregion_reset
# SR_TMP >


# SR_TMP <
cdef cRegion* cregions_store_get_new_region(cRegionsStore* store)
cdef void cregions_store_reset(cRegionsStore* store)
cdef void cregions_store_extend(cRegionsStore* store)
cdef void cregions_store_cleanup(cRegionsStore* store)
# SR_TMP >
