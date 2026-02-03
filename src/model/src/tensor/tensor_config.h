#ifndef EINSUM_IR_MODEL_TENSOR_TENSOR_CONFIG_H
#define EINSUM_IR_MODEL_TENSOR_TENSOR_CONFIG_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "common/common.h"
#include "tensor_types.h"

namespace einsum_ir::model::tensor {

  /**
   * Configuration for a binary tensor contraction operation.
   *
   * Example usage (equivalent to Python example):
   *   TensorOperationConfig config;
   *   config.primitive  = prim_t::GEMM;
   *   config.dim_types  = {dim_t::M, dim_t::M, dim_t::M, dim_t::M, dim_t::N, dim_t::K};
   *   config.exec_types = {exec_t::SEQ, exec_t::SEQ, exec_t::SEQ, exec_t::PRIM, exec_t::PRIM, exec_t::PRIM};
   *   config.dim_sizes  = {48, 36, 36, 48, 24, 36};
   *   config.strides.in0 = {2239488, 62208, 1728, 1, 0, 48};
   *   config.strides.in1 = {0, 0, 0, 0, 36, 1};
   *   config.strides.out = {1492992, 41472, 48, 1, 1728, 0};
   *   config.arch = common::Model::ZEN5;
   */
  struct TensorOperationConfig {
    prim_t primitive = prim_t::GEMM;
    std::vector<dim_t> dim_types;
    std::vector<exec_t> exec_types;
    std::vector<int64_t> dim_sizes;
    Strides strides;
    common::Model arch = common::Model::ZEN5;

    // Data types for tensors and computation
    data_t dtype_in = data_t::FP32;    // input data type
    data_t dtype_out = data_t::FP32;   // output data type
    data_t dtype_comp = data_t::FP32;  // compute data type (accumulator)
  };

  /**
   * Validation result with error message.
   */
  struct ValidationResult {
    bool valid = true;
    std::string message = "";
  };

  /**
   * Validate a tensor operation configuration.
   *
   * Checks:
   * - All vectors have the same size
   * - Stride vectors match dimension count
   * - Dimension sizes are positive
   * - At least one dimension exists
   * - Primitive is valid (GEMM)
   *
   * @param config The configuration to validate.
   * @return ValidationResult with valid flag and error message if invalid.
   */
  ValidationResult validate_config(const TensorOperationConfig& config);

  /**
   * Get the number of dimensions in the configuration.
   *
   * @param config The configuration.
   * @return Number of dimensions.
   */
  inline size_t get_num_dims(const TensorOperationConfig& config) {
    return config.dim_types.size();
  }

  /**
   * Compute the product of sizes for dimensions with a specific type.
   *
   * @param config The configuration.
   * @param type The dimension type.
   * @return Product of sizes for matching dimensions.
   */
  int64_t compute_size_product_by_dim_type(const TensorOperationConfig& config,
                                           dim_t type);

  /**
   * Get the primitive M, N, K sizes from the configuration.
   * Only considers dimensions with exec_type == PRIM.
   *
   * @param config The configuration.
   * @param o_m Output: product of M dimension sizes with PRIM exec.
   * @param o_n Output: product of N dimension sizes with PRIM exec.
   * @param o_k Output: product of K dimension sizes with PRIM exec.
   */
  void get_prim_mnk(const TensorOperationConfig& config,
                    int64_t& o_m,
                    int64_t& o_n,
                    int64_t& o_k);

  /**
   * Get the GEMM Transpose flags based on strides.
   *
   * @param config The tensor operation configuration
   * @param o_trans_a Output: transpose flag for matrix
   * @param o_trans_b Output: transpose flag for matrix B
   */
  void get_gemm_transpose_flags(const TensorOperationConfig& config,
                                int32_t& o_trans_a,
                                int32_t& o_trans_b);

  /**
   * Get the loop iteration count (product of SEQ and SHARED dimensions).
   *
   * @param config The configuration.
   * @return Total number of loop iterations.
   */
  int64_t get_loop_iterations(const TensorOperationConfig& config);

  /**
   * Get the size of all 3 Tensors (in0, in1, out) in elements.
   *
   * @param config The configuration.
   */
  double get_total_tensor_sizes(const TensorOperationConfig& config);

  /**
   * Print configuration to output stream for debugging.
   *
   * @param os Output stream.
   * @param config The configuration to print.
   */
  void print_config(std::ostream& os, const TensorOperationConfig& config);

  /**
   * Create a simple GEMM configuration (no outer loops).
   * Convenience function for testing.
   *
   * @param m The M dimension size.
   * @param n The N dimension size.
   * @param k The K dimension size.
   * @param arch The target architecture.
   * @return A TensorOperationConfig for a simple M x N x K GEMM.
   */
  TensorOperationConfig create_simple_gemm_config(int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  common::Model arch);

}  // namespace einsum_ir::model::tensor

#endif  // EINSUM_IR_MODEL_TENSOR_TENSOR_CONFIG_H
