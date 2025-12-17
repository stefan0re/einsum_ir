#include "common.h"

namespace einsum_ir::model::common {

  double get_time_model(int i_m,
                        int i_n,
                        int i_k,
                        int i_trans_a,
                        int i_trans_b,
                        Model i_model,
                        double i_peak_gflops,
                        int i_vector_size) {
    double l_gflops = 1.0;
    switch (i_model) {
      case Model::ZEN5:
        l_gflops = einsum_ir::model::zen5::get_interpolated_gflops(i_m, i_n, i_k, i_trans_a, i_trans_b);
        break;
      case Model::M4:
        l_gflops = einsum_ir::model::m4::get_interpolated_gflops(i_m, i_n, i_k, i_trans_b);
        break;
      case Model::A76:
        l_gflops = einsum_ir::model::a76::get_interpolated_gflops(i_m, i_n, i_k, i_trans_a, i_trans_b);
        break;
      case Model::GENERIC:
        l_gflops = einsum_ir::model::generic::get_interpolated_gflops(i_m, i_n, i_k, i_trans_a, i_trans_b, i_peak_gflops, i_vector_size);
        break;
    }
    std::cout << "Model GFLOPS: " << l_gflops << std::endl;
    double l_time = ((double)(i_m) * (double)(i_n) * (double)(i_k) * 2.0) / (l_gflops * 1.0e9);
    return l_time;
  }

  double get_time_xsmm(int i_m,
                       int i_n,
                       int i_k,
                       int i_trans_a,
                       int i_trans_b) {
    float* l_a;
    float* l_b;
    float* l_c;
    float* l_c_ref;

    char l_trans_a = (i_trans_a == 0) ? 'N' : 'T';
    char l_trans_b = (i_trans_b == 0) ? 'N' : 'T';

    posix_memalign((void**)&l_a, 128, i_m * i_k * sizeof(float));
    posix_memalign((void**)&l_b, 128, i_k * i_n * sizeof(float));
    posix_memalign((void**)&l_c, 128, i_m * i_n * sizeof(float));
    posix_memalign((void**)&l_c_ref, 128, i_m * i_n * sizeof(float));

    libxsmm_gemm_shape l_shape_gemm;
    libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS(l_trans_a, l_trans_b);
    libxsmm_bitfield l_prefetch_flags_brgemm = 0;

    l_shape_gemm = libxsmm_create_gemm_shape(i_m,
                                             i_n,
                                             i_k,
                                             (i_trans_a == 0) ? i_m : i_k,
                                             (i_trans_b == 0) ? i_k : i_n,
                                             i_m,
                                             libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                             libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                             libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                             libxsmm_datatype::LIBXSMM_DATATYPE_F32);

    libxsmm_gemm_batch_reduce_config l_config;
    l_config.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
    l_config.br_stride_a_hint = 0;
    l_config.br_stride_b_hint = 0;
    l_config.br_unroll_hint = 0;

    libxsmm_xmmfunction l_xmm_gemm_beta_1;
    l_xmm_gemm_beta_1.gemm = libxsmm_dispatch_brgemm(l_shape_gemm,
                                                     l_flags_brgemm,
                                                     l_prefetch_flags_brgemm,
                                                     l_config);

    std::random_device l_rd;
    std::mt19937 l_gen(l_rd());
    std::normal_distribution<float> l_dist(0.0, 1.0);

    for (int64_t l_en = 0; l_en < i_m * i_k; l_en++) {
      l_a[l_en] = l_dist(l_gen);
    }

    for (int64_t l_en = 0; l_en < i_k * i_n; l_en++) {
      l_b[l_en] = l_dist(l_gen);
    }

    for (int64_t l_en = 0; l_en < i_m * i_n; l_en++) {
      l_c[l_en] = 0.0f;
      l_c_ref[l_en] = 0.0f;
    }

    // warmup run
    size_t l_reps_warmup = 10;
    auto l_start_warmup = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < l_reps_warmup; ++i) {
      libxsmm_gemm_param l_param;
      l_param.a.primary = l_a;
      l_param.b.primary = l_b;
      l_param.c.primary = l_c;
      l_xmm_gemm_beta_1.gemm(&l_param);
    }
    auto l_end_warmup = std::chrono::high_resolution_clock::now();

    double l_warmup_duration = std::chrono::duration<double>(l_end_warmup - l_start_warmup).count();
    double l_time_per_iter = l_warmup_duration / l_reps_warmup;
    size_t l_reps = (size_t)(8.0 / l_time_per_iter);
    if (l_reps < 1) l_reps = 1;

    auto l_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < l_reps; ++i) {
      libxsmm_gemm_param l_param;
      l_param.a.primary = l_a;
      l_param.b.primary = l_b;
      l_param.c.primary = l_c;
      l_xmm_gemm_beta_1.gemm(&l_param);
    }
    auto l_end = std::chrono::high_resolution_clock::now();

    double l_duration = std::chrono::duration<double>(l_end - l_start).count();

    double l_gflops = (2.0 * i_m * i_n * i_k * l_reps) / (l_duration * 1.0e9);

    std::cout << "XSMM GFLOPS: " << l_gflops << std::endl;

    double l_time = l_duration / l_reps;

    free(l_a);
    free(l_b);
    free(l_c);
    free(l_c_ref);

    return l_time;
  }

}  // namespace einsum_ir::model::common
