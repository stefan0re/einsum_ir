#ifndef EINSUM_IR_PY_MODEL_H
#define EINSUM_IR_PY_MODEL_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/common.h"

namespace einsum_ir {
  namespace py {

    /// performance model type
    enum class model_t : uint32_t {
      zen5 = 0,
      m4 = 1,
      a76 = 2,
      generic = 3
    };

    /**
     * Performance prediction model for tensor operations.
     *
     * This class provides performance predictions for GEMM/BRGEMM operation.
     * It extracts the primitive dimensions (M, N, K, BR) and transpose flags
     * directly from the configuration parameters.
     */
    class Model {
     public:
      /// execution type
      enum class exec_t : uint32_t {
        seq = 0,
        prim = 1,
        shared = 2,
        sfc = 3,
        undefined = 99
      };

      /// primitive type
      enum class prim_t : uint32_t {
        none = 0,
        zero = 1,
        copy = 2,
        relu = 3,
        gemm = 4,
        brgemm = 5,
        undefined = 99
      };

      /// dimension type
      enum class dim_t : uint32_t {
        c = 0,
        m = 1,
        n = 2,
        k = 3,
        undefined = 99
      };

      /// data type
      enum class dtype_t : uint32_t {
        fp32 = 0,
        fp64 = 1
      };

      /**
       * Construct a Model from tensor operation configuration.
       *
       * @param prim_main The main primitive type (gemm or brgemm).
       * @param dim_types Dimension types for each dimension.
       * @param exec_types Execution types for each dimension.
       * @param dim_sizes Sizes of each dimension.
       * @param strides 3D stride tensor.
       * @param dtype The data type (fp32 or fp64).
       * @param model_type The performance model to use.
       * @param peak_gflops Peak GFLOPS for generic model (required if model_type is generic).
       * @param vector_size Vector width for generic model (required if model_type is generic).
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
