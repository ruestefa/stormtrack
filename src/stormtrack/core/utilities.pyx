# !/usr/bin/env python3

from __future__ import print_function

# C: C-libraries
from libc.math cimport pow
from libc.math cimport sqrt
from libc.stdlib cimport exit
from libc.stdlib cimport free
from libc.stdlib cimport malloc

# C: Third-party
cimport cython
cimport numpy as np
from cython.parallel cimport prange

# Standard library
import datetime as dt
import logging as log
import os
import re
import sys
import unicodedata
from pprint import pformat
from pprint import pprint

# Third-party
import numpy as np


# :call: > --- CALLERS ---
# :call: > identification::Feature::__cinit__
# :call: > identification::Feature::__repr__
# :call: > identification::Feature::unset_track
# MAX_F32 = np.finfo(np.float32).max
# MAX_F64 = np.finfo(np.float64).max
# MAX_I16 = np.iinfo(np.int16).max
# MAX_I32 = np.iinfo(np.int32).max
# MAX_I64 = np.iinfo(np.int64).max
# MAX_UI8 = np.iinfo(np.uint8).max
# MAX_UI16 = np.iinfo(np.uint16).max
# MAX_UI32 = np.iinfo(np.uint32).max
MAX_UI64 = np.iinfo(np.uint64).max
# NAN_F32 = MAX_F32
# NAN_F64 = MAX_F64
# NAN_I16 = MAX_I16
# NAN_I32 = MAX_I32
# NAN_I64 = MAX_I64
# NAN_UI8 = MAX_UI8
# NAN_UI16 = MAX_UI16
# NAN_UI32 = MAX_UI32
NAN_UI64 = MAX_UI64


# cdef void check_f32(np.float32_t fact, np.float32_t val) except *:
#     if fact <= 0 or fact >= 1:
#         raise ValueError(f"fact not in (0..1): {fact}")
#     if abs(val) > fact*MAX_F32:
#         raise Exception(f"number too big: {val:,} >= {fact:,}*{MAX_F32:,}")


# cdef void check_f64(np.float32_t fact, np.float64_t val) except *:
#     if fact <= 0 or fact >= 1:
#         raise ValueError(f"fact not in (0..1): {fact}")
#     if abs(val) > fact*MAX_F64:
#         raise Exception(f"number too big: {val:,} >= {fact:,} * {MAX_F64:,}")


# cdef void check_i32(np.float32_t fact, np.int32_t val) except *:
#     if fact <= 0 or fact >= 1:
#         raise ValueError(f"fact not in (0..1): {fact}")
#     if abs(val) > fact*MAX_I32:
#         raise Exception(f"number too big: {val:,} >= {fact:,} * {MAX_I32:,}")


# cdef void check_i64(np.float32_t fact, np.int64_t val) except *:
#     if fact <= 0 or fact >= 1:
#         raise ValueError(f"fact not in (0..1): {fact}")
#     if abs(val) > fact*MAX_I64:
#         raise Exception(f"number too big: {val:,} >= {fact:,} * {MAX_I64:,}")
