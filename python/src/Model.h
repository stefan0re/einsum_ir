#ifndef EINSUM_IR_PY_MODEL_H
#define EINSUM_IR_PY_MODEL_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "Types.h"
#include "common/common.h"

namespace einsum_ir {
  namespace py {

    /**
     * Performance prediction model for tensor operations.
     *
     * This class provides performance predictions for GEMM/BRGEMM operations
     * without requiring the full TensorOperation backend to be set up.
     * It extracts the primitive dimensions (M, N, K, BR) and transpose flags
     * directly from the configuration parameters.
     */
    class Model {
     public:
      // Use the shared enum types from Types.h
      using dim_t = einsum_ir::py::dim_t;
      using exec_t = einsum_ir::py::exec_t;
      using prim_t = einsum_ir::py::prim_t;
      using model_t = einsum_ir::py::model_t;
      using dtype_t = einsum_ir::py::dtype_t;

      /**
       * Construct a Model from tensor operation configuration.
       *
       * @param prim_main The main primitive type (gemm or brgemm).
       * @param dim_types Dimension types for each dimension.
       * @param exec_types Execution types for each dimension.
       * @param dim_sizes Sizes of each dimension.
       * @param strides 3D stride tensor [LEVEL][TENSOR][DIMENSION].
       * @param dtype The data type (fp32 or fp64).
       * @param model_type The performance model to use.
       * @param peak_gflops Peak GFLOPS for generic model (required if model_type is generic).
       * @param vector_size Vector width for generic model (required if model_type is generic).
       *
       * @throws std::invalid_argument If configuration is invalid.
       */
      Model(prim_t prim_main,
            std::vector<dim_t> const& dim_types,
            std::vector<exec_t> const& exec_types,
            std::vector<int64_t> const& dim_sizes,
            std::vector<std::vector<std::vector<int64_t>>> const& strides,
            dtype_t dtype = dtype_t::fp32,
            model_t model_type = model_t::generic,
            double peak_gflops = 0.0,
            int vector_size = 0);

      /**
       * Predict the execution time for the tensor operation.
       *
       * @return Estimated execution time in seconds.
       */
      double predict() const;

      /**
       * Predict the GFLOPS for a single GEMM operation.
       *
       * @return Estimated GFLOPS for one GEMM iteration.
       */
      double predict_gflops() const;

      // Getters for extracted parameters
      int64_t get_m() const { return m_m; }
      int64_t get_n() const { return m_n; }
      int64_t get_k() const { return m_k; }
      int64_t get_br() const { return m_br; }
      bool get_trans_a() const { return m_trans_a; }
      bool get_trans_b() const { return m_trans_b; }
      int64_t get_gemm_iter() const { return m_gemm_iter; }

     private:
      /**
       * Extract M, N, K, BR dimensions from configuration.
       */
      void extract_primitive_dims();

      /**
       * Extract transpose flags from stride patterns.
       */
      void extract_transpose_flags();

      /**
       * Compute number of GEMM iterations.
       */
      void compute_gemm_iter();

      /**
       * Convert model_t to common::Model.
       */
      einsum_ir::model::common::Model convert_model_type() const;

      /**
       * Convert dtype_t to common::DType.
       */
      einsum_ir::model::common::DType convert_dtype() const;

      // Configuration
      prim_t m_prim_main;
      std::vector<dim_t> m_dim_types;
      std::vector<exec_t> m_exec_types;
      std::vector<int64_t> m_dim_sizes;
      std::vector<std::vector<std::vector<int64_t>>> m_strides;
      dtype_t m_dtype;
      model_t m_model_type;
      double m_peak_gflops;
      int m_vector_size;

      // Extracted parameters
      int64_t m_m;
      int64_t m_n;
      int64_t m_k;
      int64_t m_br;
      bool m_trans_a;
      bool m_trans_b;
      int64_t m_gemm_iter;

      // Indices for M, N, K dimensions
      int64_t m_m_idx;
      int64_t m_n_idx;
      int64_t m_k_idx;
    };

  }  // namespace py
}  // namespace einsum_ir

#endif  // EINSUM_IR_PY_MODEL_H
