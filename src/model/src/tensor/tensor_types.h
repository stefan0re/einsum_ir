#ifndef EINSUM_IR_MODEL_TENSOR_TENSOR_TYPES_H
#define EINSUM_IR_MODEL_TENSOR_TENSOR_TYPES_H

#include <array>
#include <cstdint>
#include <vector>

namespace einsum_ir::model::tensor {

  /**
   * Primitive operation type.
   * Currently only GEMM is supported.
   */
  enum class prim_t {
    GEMM = 0,
    UNDEFINED_PRIM = 99
  };

  /**
   * Dimension type for tensor contractions.
   * - M: appears in left input and output
   * - N: appears in right input and output
   * - K: appears in left and right input (contracted)
   * - C: batch dimension, appears in all tensors
   */
  enum class dim_t {
    M = 0,
    N = 1,
    K = 2,
    C = 3,
    UNDEFINED_DIM = 99
  };

  /**
   * Execution type for each dimension.
   * - SEQ:    sequential execution
   * - SHARED: shared/parallel execution (e.g., OpenMP)
   * - PRIM:   handled by the primitive (GEMM kernel)
   */
  enum class exec_t {
    SEQ = 0,
    SHARED = 1,
    PRIM = 2,
    UNDEFINED_EXEC = 99
  };

  /**
   * Data type for tensor elements.
   */
  enum class data_t {
    FP32 = 0,   // 32-bit floating point (float)
    FP64 = 1,   // 64-bit floating point (double)
    FP16 = 2,   // 16-bit floating point (half)
    BF16 = 3,   // bfloat16
    INT32 = 4,  // 32-bit integer
    INT64 = 5,  // 64-bit integer
    UNDEFINED_DTYPE = 99
  };

  /**
   * Get the size in bytes for a data type.
   *
   * @param dtype The data type.
   * @return Size in bytes, or 0 for undefined.
   */
  inline int64_t dtype_size(data_t dtype) {
    switch (dtype) {
      case data_t::FP32:
        return 4;
      case data_t::FP64:
        return 8;
      case data_t::FP16:
        return 2;
      case data_t::BF16:
        return 2;
      case data_t::INT32:
        return 4;
      case data_t::INT64:
        return 8;
      default:
        return 0;
    }
  }

  /**
   * Strides for a single tensor.
   * Each element corresponds to a dimension in the dim_types vector.
   */
  using TensorStrides = std::vector<int64_t>;

  /**
   * Strides for all three tensors in a binary contraction.
   * Index 0: input tensor 0 (left)
   * Index 1: input tensor 1 (right)
   * Index 2: output tensor
   */
  struct Strides {
    TensorStrides in0;  // left input strides
    TensorStrides in1;  // right input strides
    TensorStrides out;  // output strides
  };

  /**
   * Convert prim_t to string for debugging/logging.
   */
  inline const char* prim_to_string(prim_t p) {
    switch (p) {
      case prim_t::GEMM:
        return "GEMM";
      case prim_t::UNDEFINED_PRIM:
        return "UNDEFINED_PRIM";
      default:
        return "UNKNOWN";
    }
  }

  /**
   * Convert dim_t to string for debugging/logging.
   */
  inline const char* dim_to_string(dim_t d) {
    switch (d) {
      case dim_t::M:
        return "M";
      case dim_t::N:
        return "N";
      case dim_t::K:
        return "K";
      case dim_t::C:
        return "C";
      case dim_t::UNDEFINED_DIM:
        return "UNDEFINED_DIM";
      default:
        return "UNKNOWN";
    }
  }

  /**
   * Convert exec_t to string for debugging/logging.
   */
  inline const char* exec_to_string(exec_t e) {
    switch (e) {
      case exec_t::SEQ:
        return "SEQ";
      case exec_t::SHARED:
        return "SHARED";
      case exec_t::PRIM:
        return "PRIM";
      case exec_t::UNDEFINED_EXEC:
        return "UNDEFINED_EXEC";
      default:
        return "UNKNOWN";
    }
  }

  /**
   * Convert data_t to string for debugging/logging.
   */
  inline const char* dtype_to_string(data_t d) {
    switch (d) {
      case data_t::FP32:
        return "FP32";
      case data_t::FP64:
        return "FP64";
      case data_t::FP16:
        return "FP16";
      case data_t::BF16:
        return "BF16";
      case data_t::INT32:
        return "INT32";
      case data_t::INT64:
        return "INT64";
      case data_t::UNDEFINED_DTYPE:
        return "UNDEFINED_DTYPE";
      default:
        return "UNKNOWN";
    }
  }

}  // namespace einsum_ir::model::tensor

#endif  // EINSUM_IR_MODEL_TENSOR_TENSOR_TYPES_H
