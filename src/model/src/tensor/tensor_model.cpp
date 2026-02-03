#include "tensor_model.h"

namespace einsum_ir::model::tensor {

  ModelResult estimate_performance(const TensorOperationConfig& config) {
    ModelResult result;

    result.flops = compute_total_flops(config);
    result.time_seconds = get_kernel_time(config) * get_loop_iterations(config);
    result.gflops = result.flops / (result.time_seconds * 1.0e9);
    result.efficiency = 0.0;
    result.memory_bytes = get_total_tensor_sizes(config);

    return result;
  }

  double estimate_time(const TensorOperationConfig& config) {
    ModelResult result = estimate_performance(config);
    return result.time_seconds;
  }

  double estimate_gflops(const TensorOperationConfig& config) {
    ModelResult result = estimate_performance(config);
    return result.gflops;
  }

  int64_t compute_total_flops(const TensorOperationConfig& config) {
    int64_t total_m = compute_size_product_by_dim_type(config, dim_t::M);
    int64_t total_n = compute_size_product_by_dim_type(config, dim_t::N);
    int64_t total_k = compute_size_product_by_dim_type(config, dim_t::K);
    int64_t total_c = compute_size_product_by_dim_type(config, dim_t::C);

    return total_m * total_n * (2 * total_k - 1) * total_c;
  }

  int64_t compute_memory_traffic(const TensorOperationConfig& config,
                                 const HardwareConstants& hw) {
    (void)config;
    (void)hw;
    return 0;
  }

  double compute_compute_time(const TensorOperationConfig& config,
                              const HardwareConstants& hw) {
    (void)config;
    (void)hw;
    return 0.0;
  }

  double compute_memory_time(const TensorOperationConfig& config,
                             const HardwareConstants& hw) {
    (void)config;
    (void)hw;
    return 0.0;
  }

  void analyze_data_reuse(const TensorOperationConfig& config,
                          double& o_reuse_in0,
                          double& o_reuse_in1,
                          double& o_reuse_out) {
    (void)config;
    o_reuse_in0 = 1.0;
    o_reuse_in1 = 1.0;
    o_reuse_out = 1.0;
  }

  double get_kernel_time(const TensorOperationConfig& config) {
    int64_t prim_m, prim_n, prim_k;
    get_prim_mnk(config, prim_m, prim_n, prim_k);
    int32_t trans_a, trans_b;
    get_gemm_transpose_flags(config, trans_a, trans_b);

    double gflops = 0.0;
    double time = 0.0;
    if (config.arch != common::Model::GENERIC) {
      time = common::get_time_model(static_cast<int>(prim_m),
                                    static_cast<int>(prim_n),
                                    static_cast<int>(prim_k),
                                    trans_a, trans_b,
                                    config.arch,
                                    gflops);
    } else {
      time = common::get_time_model(static_cast<int>(prim_m),
                                    static_cast<int>(prim_n),
                                    static_cast<int>(prim_k),
                                    0, 0,
                                    config.arch,
                                    gflops,
                                    100,
                                    16);
    }
    return time;
  }
}  // namespace einsum_ir::model::tensor
