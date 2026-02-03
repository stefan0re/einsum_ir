#ifndef EINSUM_IR_MODEL_TENSOR_TENSOR_MODEL_H
#define EINSUM_IR_MODEL_TENSOR_TENSOR_MODEL_H

#include "tensor_config.h"
#include "tensor_constants.h"

namespace einsum_ir::model::tensor {

  /**
   * Result of performance model estimation.
   */
  struct ModelResult {
    double time_seconds;   // Estimated execution time in seconds
    double gflops;         // Estimated GFLOPS achieved
    double efficiency;     // Efficiency (0.0 - 1.0) relative to peak
    int64_t flops;         // Total floating point operations
    int64_t memory_bytes;  // Memory traffic in bytes
  };

  //============================================================================
  // Main Interface - Implement these functions
  //============================================================================

  /**
   * Estimate performance for a tensor contraction operation.
   *
   * This is the main entry point for the performance model.
   * Implement your performance model logic here.
   *
   * @param config The tensor operation configuration.
   * @return ModelResult with estimated performance metrics.
   */
  ModelResult estimate_performance(const TensorOperationConfig& config);

  /**
   * Estimate execution time for a tensor contraction operation.
   *
   * @param config The tensor operation configuration.
   * @return Estimated execution time in seconds.
   */
  double estimate_time(const TensorOperationConfig& config);

  /**
   * Estimate GFLOPS for a tensor contraction operation.
   *
   * @param config The tensor operation configuration.
   * @return Estimated GFLOPS.
   */
  double estimate_gflops(const TensorOperationConfig& config);

  //============================================================================
  // Helper Functions - Implement as needed for your model
  //============================================================================

  /**
   * Compute total FLOPs for the tensor contraction.
   * For GEMM: 2 * M * N * K (multiply-add).
   *
   * @param config The tensor operation configuration.
   * @return Total floating point operations.
   */
  int64_t compute_total_flops(const TensorOperationConfig& config);

  /**
   * Compute memory traffic for the tensor contraction.
   * Consider data reuse based on loop structure and cache sizes.
   *
   * @param config The tensor operation configuration.
   * @param hw Hardware constants for cache modeling.
   * @return Estimated memory traffic in bytes.
   */
  int64_t compute_memory_traffic(const TensorOperationConfig& config,
                                 const HardwareConstants& hw);

  /**
   * Compute the time spent on computation (compute-bound estimate).
   *
   * @param config The tensor operation configuration.
   * @param hw Hardware constants.
   * @return Estimated compute time in seconds.
   */
  double compute_compute_time(const TensorOperationConfig& config,
                              const HardwareConstants& hw);

  /**
   * Compute the time spent on memory transfers (memory-bound estimate).
   *
   * @param config The tensor operation configuration.
   * @param hw Hardware constants.
   * @return Estimated memory time in seconds.
   */
  double compute_memory_time(const TensorOperationConfig& config,
                             const HardwareConstants& hw);

  /**
   * Analyze the loop structure to determine data reuse patterns.
   * This can help identify which tensors are reused across iterations.
   *
   * @param config The tensor operation configuration.
   * @param o_reuse_in0 Output: reuse factor for input 0.
   * @param o_reuse_in1 Output: reuse factor for input 1.
   * @param o_reuse_out Output: reuse factor for output.
   */
  void analyze_data_reuse(const TensorOperationConfig& config,
                          double& o_reuse_in0,
                          double& o_reuse_in1,
                          double& o_reuse_out);

  /**
   * Get the GEMM kernel efficiency based on M, N, K dimensions.
   * Uses the existing architecture-specific models.
   *
   * @param config The tensor operation configuration.
   * @return Estimated GFLOPS for the primitive GEMM kernel.
   */
  double get_kernel_time(const TensorOperationConfig& config);

}  // namespace einsum_ir::model::tensor

#endif  // EINSUM_IR_MODEL_TENSOR_TENSOR_MODEL_H
