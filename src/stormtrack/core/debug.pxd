# -*- coding: utf-8 -*-

# Local
from .structs cimport cPixel
from .structs cimport cRegion


cdef void cregion_dump(cRegion* cregion, str path) except *
