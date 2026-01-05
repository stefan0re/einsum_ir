#include <iomanip>
#include <iostream>

#include "tensor/frontend.h"

using namespace einsum_ir::model::tensor::frontend;
using namespace einsum_ir::model::common;

int main() {
  // Simple matrix multiplication C = A * B
  // A: 1024 x 512 (M x K)
  // B: 512 x 256 (K x N)
  // C: 1024 x 256 (M x N)
  //
  // Using primitive sizes of 32 x 32 x 32 for M x N x K

  std::cout << "=== Tensor Operation Frontend Example ===" << std::endl;
  std::cout << "\nExample 1: Matrix Multiplication C = A * B" << std::endl;
  std::cout << "Matrix A: 1024 x 512" << std::endl;
  std::cout << "Matrix B: 512 x 256" << std::endl;
  std::cout << "Matrix C: 1024 x 256" << std::endl;

  TensorOperationConfig config;

  std::vector<DimensionType> dim_types = {
      DimensionType::m,
      DimensionType::n,
      DimensionType::k,
      DimensionType::m,
      DimensionType::n,
      DimensionType::k};

  std::vector<ExecutionType> exec_types = {
      ExecutionType::seq,
      ExecutionType::seq,
      ExecutionType::seq,
      ExecutionType::prim,
      ExecutionType::prim,
      ExecutionType::prim};

  std::vector<int64_t> dim_sizes = {32, 8, 16, 32, 32, 32};
  std::vector<int64_t> strides_in0 = {32, 0, 1024 * 32, 1, 0, 1024};
  std::vector<int64_t> strides_in1 = {0, 32 * 512, 32, 0, 512, 1};
  std::vector<int64_t> strides_out = {32, 1024 * 32, 0, 1, 1024, 0};

  config.setup(
      DataType::float32,
      PrimitiveType::zero,
      PrimitiveType::gemm,
      PrimitiveType::none,
      dim_types,
      exec_types,
      dim_sizes,
      strides_in0,
      strides_in1,
      strides_out,
      Model::GENERIC,
      100.0,
      8);

  std::cout << "\nConfiguration Details:" << std::endl;
  std::cout << "  Primitive M size: " << config.get_prim_m_size() << std::endl;
  std::cout << "  Primitive N size: " << config.get_prim_n_size() << std::endl;
  std::cout << "  Primitive K size: " << config.get_prim_k_size() << std::endl;
  std::cout << "  Loop iterations: " << config.get_loop_iterations() << std::endl;
  std::cout << "  Transpose A: " << (config.get_trans_a() ? "Yes" : "No") << std::endl;
  std::cout << "  Transpose B: " << (config.get_trans_b() ? "Yes" : "No") << std::endl;

  std::cout << "\n=== Example Complete ===" << std::endl;

  return 0;
}
