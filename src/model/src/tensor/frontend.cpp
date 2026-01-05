#include "frontend.h"

#include <iostream>

namespace einsum_ir::model::tensor::frontend {

  TensorOperationConfig::TensorOperationConfig()
      : m_data_type(DataType::undefined),
        m_prim_first(PrimitiveType::undefined),
        m_prim_main(PrimitiveType::undefined),
        m_prim_last(PrimitiveType::undefined),
        m_num_dims(0),
        m_prim_m_size(0),
        m_prim_n_size(0),
        m_prim_k_size(0),
        m_prim_br_size(0),
        m_model(common::Model::GENERIC),
        m_peak_gflops(0.0),
        m_vector_size(0),
        m_trans_a(false),
        m_trans_b(false),
        m_loop_iterations(1) {
  }

  void TensorOperationConfig::setup(
      DataType i_data_type,
      PrimitiveType i_prim_first,
      PrimitiveType i_prim_main,
      PrimitiveType i_prim_last,
      std::vector<DimensionType> const& i_dim_types,
      std::vector<ExecutionType> const& i_exec_types,
      std::vector<int64_t> const& i_dim_sizes,
      std::vector<int64_t> const& i_strides_in0,
      std::vector<int64_t> const& i_strides_in1,
      std::vector<int64_t> const& i_strides_out,
      common::Model i_model,
      double i_peak_gflops,
      int i_vector_size) {
    if (i_dim_types.size() != i_exec_types.size()) {
      std::cerr << "Error: Dimension types and execution types must have same size" << std::endl;
      return;
    }
    if (i_dim_types.size() != i_dim_sizes.size()) {
      std::cerr << "Error: Dimension types and sizes must have same size" << std::endl;
      return;
    }
    if (i_dim_types.size() != i_strides_in0.size()) {
      std::cerr << "Error: Dimension types and strides_in0 must have same size" << std::endl;
      return;
    }
    if (i_dim_types.size() != i_strides_in1.size()) {
      std::cerr << "Error: Dimension types and strides_in1 must have same size" << std::endl;
      return;
    }
    if (i_dim_types.size() != i_strides_out.size()) {
      std::cerr << "Error: Dimension types and strides_out must have same size" << std::endl;
      return;
    }

    m_data_type = i_data_type;
    m_prim_first = i_prim_first;
    m_prim_main = i_prim_main;
    m_prim_last = i_prim_last;

    m_dim_types = i_dim_types;
    m_exec_types = i_exec_types;
    m_dim_sizes = i_dim_sizes;
    m_num_dims = static_cast<int64_t>(i_dim_types.size());

    m_strides_in0 = i_strides_in0;
    m_strides_in1 = i_strides_in1;
    m_strides_out = i_strides_out;

    m_prim_m_size = 0;
    m_prim_n_size = 0;
    m_prim_k_size = 0;
    m_prim_br_size = 0;

    for (int64_t i = 0; i < m_num_dims; i++) {
      if (i_exec_types[i] == ExecutionType::prim) {
        if (i_dim_types[i] == DimensionType::m) {
          m_prim_m_size = i_dim_sizes[i];
        } else if (i_dim_types[i] == DimensionType::n) {
          m_prim_n_size = i_dim_sizes[i];
        } else if (i_dim_types[i] == DimensionType::k) {
          if (i != m_num_dims - 1) {
            m_prim_br_size = i_dim_sizes[i];
          } else {
            m_prim_k_size = i_dim_sizes[i];
          }
        }
      }
    }

    if (m_prim_m_size == 0) {
      std::cerr << "Error: Primitive M size not set" << std::endl;
      return;
    }
    if (m_prim_n_size == 0) {
      std::cerr << "Error: Primitive N size not set" << std::endl;
      return;
    }
    if (m_prim_k_size == 0) {
      std::cerr << "Error: Primitive K size not set" << std::endl;
      return;
    }

    m_model = i_model;
    m_peak_gflops = i_peak_gflops;
    m_vector_size = i_vector_size;

    determine_transpose_flags();
    calculate_loop_iterations();
  }

  void TensorOperationConfig::determine_transpose_flags() {
    m_trans_a = false;
    m_trans_b = false;

    for (int64_t i = 0; i < m_num_dims; i++) {
      if (m_exec_types[i] == ExecutionType::prim) {
        if (m_dim_types[i] == DimensionType::k && m_strides_in0[i] == 1) {
          m_trans_a = true;
        }
        if (m_dim_types[i] == DimensionType::n && m_strides_in1[i] == 1) {
          m_trans_b = true;
        }
      }
    }
  }

  void TensorOperationConfig::calculate_loop_iterations() {
    m_loop_iterations = 1;

    for (int64_t i = 0; i < m_num_dims; i++) {
      if (m_exec_types[i] != ExecutionType::prim) {
        m_loop_iterations *= m_dim_sizes[i];
      }
    }
  }

}  // namespace einsum_ir::model::tensor::frontend
