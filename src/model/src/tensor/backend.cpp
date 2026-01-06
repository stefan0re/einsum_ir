#include "backend.h"

namespace einsum_ir::model::tensor::backend {

  double compute_time(frontend::TensorOperationConfig const& config,
                      double& o_gflops) {
    int64_t m = config.get_prim_m_size();
    int64_t n = config.get_prim_n_size();
    int64_t k = config.get_prim_k_size();
    int64_t iterations = config.get_loop_iterations();

    int trans_a = config.get_trans_a() ? 1 : 0;
    int trans_b = config.get_trans_b() ? 1 : 0;

    double exec_time = common::get_time_model(
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        trans_a,
        trans_b,
        config.get_model(),
        o_gflops,
        config.get_peak_gflops(),
        config.get_vector_size());

    return exec_time * iterations;
  }

  double memory_time(frontend::TensorOperationConfig const& config,
                     double& o_gflops) {
    int64_t m = config.get_prim_m_size();
    int64_t n = config.get_prim_n_size();
    int64_t k = config.get_prim_k_size();
    int64_t iterations = config.get_loop_iterations();

    int64_t bytes_per_elem = 4;
    if (config.get_data_type() == frontend::DataType::float64) {
      bytes_per_elem = 8;
    }

    int64_t data_volume_bytes = (m * k + k * n + m * n) * bytes_per_elem * iterations;
    double data_volume_gb = data_volume_bytes / 1e9;

    double exec_time = data_volume_gb / DRAM_BANDWIDTH_GB_S;

    o_gflops = (2.0 * m * n * k * iterations) / (exec_time * 1e9);

    return exec_time;
  }

}  // namespace einsum_ir::model::tensor::backend
