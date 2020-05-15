# -*- coding: utf-8 -*-

# Local
from .structs cimport cConstants
from .structs cimport cPixel
from .structs cimport cRegion


cdef void cregion_dump(cRegion* cregion, str path, cConstants constants) except *
