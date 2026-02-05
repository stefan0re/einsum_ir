#include "tensor_model.h"

namespace einsum_ir::model::tensor {

  ModelResult estimate_performance(const TensorOperationConfig& config) {
    ModelResult result;

    analyze_data_reuse(config);

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

  void estimate_tensor_reuse(const TensorOperationConfig& config,
                             int tensor_id,
                             double& reuse_factor,
                             int& level) {
    // tenor id 0 -> in0
    // tenor id 1 -> in1
    // tenor id 2 -> out

    dim_t x = dim_t::UNDEFINED_DIM;
    switch (tensor_id) {
      case 0:
        x = dim_t::N;
        break;
      case 1:
        x = dim_t::M;
        break;
      case 2:
        x = dim_t::K;
        break;
      default:
        std::cerr << "Wrong Tensor ID" << std::endl;
        return;
        break;
    }

    double current_size = 4.0;  // TODO
    double iteration_todo = get_loop_iterations(config);

    for (int i = config.dim_sizes.size() - 1; i >= 0; i--) {
      std::cout << "iteration todo: " << iteration_todo << std::endl;
      if ((config.exec_types[i] != exec_t::PRIM) && config.dim_types[i] == x) {
        reuse_factor = 1.0 - (1.0 / config.dim_sizes[i]);
        break;
      }
      if (config.dim_types[i] != x) {
        current_size *= config.dim_sizes[i];
      }
      if (config.exec_types[i] == exec_t::SEQ) {
        iteration_todo /= config.dim_sizes[i];
      }
    }

    if (current_size < 1024 * 1024 * 8) {  // currently M4
      level = 2;
    }

    std::cout << "------------------------" << std::endl;
  }

  void analyze_data_reuse(const TensorOperationConfig& config) {
    double re_in0, re_in1, re_out = 0.0;
    int lvl_in0, lvl_in1, lvl_out = 0;

    estimate_tensor_reuse(config, 0, re_in0, lvl_in0);
    estimate_tensor_reuse(config, 1, re_in1, lvl_in1);
    estimate_tensor_reuse(config, 2, re_out, lvl_out);

    std::cout << "IN0: " << std::endl;
    std::cout << "  Reuse ratio: " << re_in0 << std::endl;
    std::cout << "  Reuse level: " << lvl_in0 << std::endl;

    std::cout << "IN1: " << std::endl;
    std::cout << "  Reuse ratio: " << re_in1 << std::endl;
    std::cout << "  Reuse level: " << lvl_in1 << std::endl;

    std::cout << "OUT: " << std::endl;
    std::cout << "  Reuse ratio: " << re_out << std::endl;
    std::cout << "  Reuse level: " << lvl_out << std::endl;
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
