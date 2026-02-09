#include "Model.h"

namespace einsum_ir {
  namespace py {

    Model::Model(prim_t prim_main,
                 std::vector<dim_t> const& dim_types,
                 std::vector<exec_t> const& exec_types,
                 std::vector<int64_t> const& dim_sizes,
                 std::vector<std::vector<std::vector<int64_t>>> const& strides,
                 dtype_t dtype,
                 model_t model_type,
                 double peak_gflops,
                 int vector_size)
        : m_prim_main(prim_main),
          m_dim_types(dim_types),
          m_exec_types(exec_types),
          m_dim_sizes(dim_sizes),
          m_strides(strides),
          m_dtype(dtype),
          m_model_type(model_type),
          m_peak_gflops(peak_gflops),
          m_vector_size(vector_size),
          m_m(0),
          m_n(0),
          m_k(0),
          m_br(1),
          m_trans_a(false),
          m_trans_b(false),
          m_gemm_iter(1),
          m_m_idx(-1),
          m_n_idx(-1),
          m_k_idx(-1) {


      extract_primitive_dims();
      extract_transpose_flags();
      compute_gemm_iter();
    }

    void Model::extract_primitive_dims() {

      std::vector<int64_t> prim_dims;
      for (size_t i = 0; i < m_dim_types.size(); i++) {
        if (m_exec_types[i] == exec_t::prim) {
          prim_dims.push_back(m_dim_sizes[i]);
        }
      }

      bool is_brgemm = (m_prim_main == prim_t::brgemm);
      size_t expected_prims = is_brgemm ? 4 : 3;


      if( is_brgemm ){
        m_br = prim_dims[0];
        m_m = prim_dims[1];
        m_n = prim_dims[2];
        m_k = prim_dims[3];
      } else {
        m_m = prim_dims[0];
        m_n = prim_dims[1];
        m_k = prim_dims[2];
        m_br = 1;
      }
    }

    void Model::extract_transpose_flags() {

      auto const& strides_a = m_strides[0][0];
      auto const& strides_b = m_strides[0][1];

      for( size_t i = 0; i < m_dim_types.size(); i++ ) {
        if( m_dim_types[i] == dim_t::m && m_exec_types[i] == exec_t::prim ) {
          m_m_idx = i;
        } else if( m_dim_types[i] == dim_t::n && m_exec_types[i] == exec_t::prim ) {
          m_n_idx = i;
        } else if( m_dim_types[i] == dim_t::k && m_exec_types[i] == exec_t::prim ) {
          m_k_idx = i;
        }
      }

      int64_t stride_a_m = strides_a[m_m_idx];
      int64_t stride_a_k = strides_a[m_k_idx];
      int64_t stride_b_k = strides_b[m_k_idx];
      int64_t stride_b_n = strides_b[m_n_idx];

      if (stride_a_k == 1) {
        m_trans_a = true;
      } else {
        m_trans_a = false;
      }

      if (stride_b_n == 1) {
        m_trans_b = true;
      } else {
        m_trans_b = false;
      }
    }

    void Model::compute_gemm_iter() {
      m_gemm_iter = 1;
      for (size_t i = 0; i < m_exec_types.size(); i++) {
        if (m_exec_types[i] != exec_t::prim) {
          m_gemm_iter *= m_dim_sizes[i];
        }
      }
    }

    einsum_ir::model::common::Model Model::convert_model_type() const {
      switch (m_model_type) {
        case model_t::zen5:
          return einsum_ir::model::common::Model::ZEN5;
        case model_t::m4:
          return einsum_ir::model::common::Model::M4;
        case model_t::a76:
          return einsum_ir::model::common::Model::A76;
        case model_t::generic:
        default:
          return einsum_ir::model::common::Model::GENERIC;
      }
    }

    einsum_ir::model::common::DType Model::convert_dtype() const {
      switch (m_dtype) {
        case dtype_t::fp32:
          return einsum_ir::model::common::DType::FP32;
        case dtype_t::fp64:
          return einsum_ir::model::common::DType::FP64;
        default:
          return einsum_ir::model::common::DType::FP32;
      }
    }

    double Model::predict() const {

      double o_gflops = 0.0;
      double time_per_gemm = einsum_ir::model::common::get_time_model(
          static_cast<int>(m_m),
          static_cast<int>(m_n),
          static_cast<int>(m_k * m_br),
          m_trans_a ? 1 : 0,
          m_trans_b ? 1 : 0,
          convert_dtype(),
          convert_model_type(),
          o_gflops,
          m_peak_gflops,
          m_vector_size);

      return time_per_gemm * static_cast<double>(m_gemm_iter);
    }

    double Model::predict_gflops() const {

      double o_gflops = 0.0;
      einsum_ir::model::common::get_time_model(
          static_cast<int>(m_m),
          static_cast<int>(m_n),
          static_cast<int>(m_k * m_br),
          m_trans_a ? 1 : 0,
          m_trans_b ? 1 : 0,
          convert_dtype(),
          convert_model_type(),
          o_gflops,
          m_peak_gflops,
          m_vector_size);

      return o_gflops;
    }

  }  // namespace py
}  // namespace einsum_ir
