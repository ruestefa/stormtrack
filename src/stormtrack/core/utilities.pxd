
cimport numpy as np

# ==============================================================================

cdef np.uint8_t   MAX_UI8
cdef np.uint16_t  MAX_UI16
cdef np.uint32_t  MAX_UI32
cdef np.uint64_t  MAX_UI64
cdef np.int16_t   MAX_I16
cdef np.int32_t   MAX_I32
cdef np.int64_t   MAX_I64
cdef np.float32_t MAX_F32
cdef np.float64_t MAX_F64

cdef np.uint8_t   NAN_UI8
cdef np.uint16_t  NAN_UI16
cdef np.uint32_t  NAN_UI32
cdef np.uint64_t  NAN_UI64
cdef np.int16_t   NAN_I16
cdef np.int32_t   NAN_I32
cdef np.int64_t   NAN_I64
cdef np.float32_t NAN_F32
cdef np.float64_t NAN_F64

cdef void check_f32(np.float32_t fact, np.float32_t val) except *
cdef void check_f64(np.float32_t fact, np.float64_t val) except *
cdef void check_i32(np.float32_t fact, np.int32_t val) except *
cdef void check_i64(np.float32_t fact, np.int64_t val) except *

# ==============================================================================
