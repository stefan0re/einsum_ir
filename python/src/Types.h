#ifndef EINSUM_IR_PY_TYPES_H
#define EINSUM_IR_PY_TYPES_H

#include <cstdint>

namespace einsum_ir {
  namespace py {

    /**
     * Shared type definitions for tensor operations.
     */

    /// execution type
    enum class exec_t : uint32_t {
      seq = 0,
      prim = 1,
      shared = 2,
      sfc = 3,
      undefined = 99
    };

    /// primitive type
    enum class prim_t : uint32_t {
      none = 0,
      zero = 1,
      copy = 2,
      relu = 3,
      gemm = 4,
      brgemm = 5,
      undefined = 99
    };

    /// dimension type
    enum class dim_t : uint32_t {
      c = 0,
      m = 1,
      n = 2,
      k = 3,
      undefined = 99
    };

    /// data type
    enum class dtype_t : uint32_t {
      fp32 = 0,
      fp64 = 1
    };

    /// error codes
    enum class error_t : int32_t {
      success = 0,
      compilation_failed = 1,
      invalid_stride_shape = 2,
      invalid_optimization_config = 3
    };

    /// performance model type
    enum class model_t : uint32_t {
      zen5 = 0,
      m4 = 1,
      a76 = 2,
      generic = 3
    };

  }  // namespace py
}  // namespace einsum_ir

#endif  // EINSUM_IR_PY_TYPES_H
