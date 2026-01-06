#ifndef EINSUM_IR_MODEL_TENSOR_FRONTEND_H
#define EINSUM_IR_MODEL_TENSOR_FRONTEND_H

#include <cstdint>
#include <tuple>
#include <vector>

#include "../common/common.h"

namespace einsum_ir::model::tensor::frontend {

  enum class DataType : uint8_t;
  enum class PrimitiveType : uint8_t;
  enum class DimensionType : uint8_t;
  enum class ExecutionType : uint8_t;

  /**
   * Enumerations for TEIR configuration
   */
  enum class DataType : uint8_t {
    float32,
    float64,
    undefined
  };

  enum class PrimitiveType : uint8_t {
    none,
    zero,
    gemm,
    add,
    copy,
    undefined
  };

  enum class DimensionType : uint8_t {
    k,
    m,
    n,
    c,
    undefined
  };

  enum class ExecutionType : uint8_t {
    seq,
    shared,
    sfc,
    prim,
    undefined
  };

  /**
   * TEIR Configuration Class
   *
   * Stores complete information about a tensor contraction operation including:
   * - Data types
   * - Primitive operations (first, main, last)
   * - Dimension types and sizes
   * - Execution strategies
   * - Memory strides for input/output tensors
   */
  class TensorOperationConfig {
   private:
    // Data type information
    DataType m_data_type;

    // Primitive operations
    PrimitiveType m_prim_first;
    PrimitiveType m_prim_main;
    PrimitiveType m_prim_last;

    // Dimension information
    std::vector<DimensionType> m_dim_types;
    std::vector<ExecutionType> m_exec_types;
    std::vector<int64_t> m_dim_sizes;

    // Stride information for tensors
    std::vector<int64_t> m_strides_in0;  // strides for first input tensor
    std::vector<int64_t> m_strides_in1;  // strides for second input tensor
    std::vector<int64_t> m_strides_out;  // strides for output tensor

    // Number of dimensions
    int64_t m_num_dims;

    int64_t m_prim_m_size;
    int64_t m_prim_n_size;
    int64_t m_prim_k_size;
    int64_t m_prim_br_size;

    // Performance modeling
    common::Model m_model;
    double m_peak_gflops;
    int m_vector_size;
    bool m_trans_a;
    bool m_trans_b;
    int64_t m_loop_iterations;

    /**
     * Determines transpose flags from strides.
     * A is transposed if in0 stride of k dimension is 1
     * B is transposed if in1 stride of n dimension is 1
     */
    void determine_transpose_flags();

    /**
     * Calculates the total number of loop iterations.
     * This is the product of all non-primitive dimension sizes.
     */
    void calculate_loop_iterations();

   public:
    /**
     * Default constructor
     */
    TensorOperationConfig();

    /**
     * Setup function to initialize all configuration parameters.
     *
     * @param i_data_type Data type for the operation (float32, float64, etc.)
     * @param i_prim_first First primitive operation (e.g., zero initialization)
     * @param i_prim_main Main primitive operation (e.g., gemm)
     * @param i_prim_last Last primitive operation (e.g., none, add)
     * @param i_dim_types Vector of dimension types (k, m, n, c, etc.)
     * @param i_exec_types Vector of execution types (seq, omp, sfc, prim)
     * @param i_dim_sizes Vector of dimension sizes
     * @param i_strides_in0 Stride vector for first input tensor
     * @param i_strides_in1 Stride vector for second input tensor
     * @param i_strides_out Stride vector for output tensor
     * @param i_model Performance model to use (default: GENERIC)
     * @param i_peak_gflops Peak GFLOPS for generic model (default: 0.0)
     * @param i_vector_size Vector width for generic model (default: 0)
     */
    void setup(DataType i_data_type,
               PrimitiveType i_prim_first,
               PrimitiveType i_prim_main,
               PrimitiveType i_prim_last,
               std::vector<DimensionType> const& i_dim_types,
               std::vector<ExecutionType> const& i_exec_types,
               std::vector<int64_t> const& i_dim_sizes,
               std::vector<int64_t> const& i_strides_in0,
               std::vector<int64_t> const& i_strides_in1,
               std::vector<int64_t> const& i_strides_out,
               common::Model i_model = common::Model::GENERIC,
               double i_peak_gflops = 0.0,
               int i_vector_size = 0);

    // Getters
    DataType get_data_type() const { return m_data_type; }
    PrimitiveType get_prim_first() const { return m_prim_first; }
    PrimitiveType get_prim_main() const { return m_prim_main; }
    PrimitiveType get_prim_last() const { return m_prim_last; }

    std::vector<DimensionType> const& get_dim_types() const { return m_dim_types; }
    std::vector<ExecutionType> const& get_exec_types() const { return m_exec_types; }
    std::vector<int64_t> const& get_dim_sizes() const { return m_dim_sizes; }

    std::vector<int64_t> const& get_strides_in0() const { return m_strides_in0; }
    std::vector<int64_t> const& get_strides_in1() const { return m_strides_in1; }
    std::vector<int64_t> const& get_strides_out() const { return m_strides_out; }

    int64_t get_num_dims() const { return m_num_dims; }
    int64_t get_prim_m_size() const { return m_prim_m_size; }
    int64_t get_prim_n_size() const { return m_prim_n_size; }
    int64_t get_prim_k_size() const { return m_prim_k_size; }
    int64_t get_prim_br_size() const { return m_prim_br_size; }
    bool get_trans_a() const { return m_trans_a; }
    bool get_trans_b() const { return m_trans_b; }
    int64_t get_loop_iterations() const { return m_loop_iterations; }
    common::Model get_model() const { return m_model; }
    double get_peak_gflops() const { return m_peak_gflops; }
    int get_vector_size() const { return m_vector_size; }
  };

}  // namespace einsum_ir::model::tensor::frontend

#endif  // EINSUM_IR_MODEL_TENSOR_FRONTEND_H