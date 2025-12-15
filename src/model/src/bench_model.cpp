#include <cstring>
#include <iostream>

#include "common/common.h"

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << "Usage: " << argv[0] << " <m> <n> <k> <trans_a> <trans_b> <model>" << std::endl;
    std::cout << "  m, n, k:    Matrix dimensions (positive integers)" << std::endl;
    std::cout << "  trans_a:    Transpose A matrix (0 or 1)" << std::endl;
    std::cout << "  trans_b:    Transpose B matrix (0 or 1)" << std::endl;
    std::cout << "  model:      Performance model to use (zen5, m4, pi5, generic)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example: " << argv[0] << " 64 48 64 0 0 zen5" << std::endl;
    return EXIT_FAILURE;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);
  int trans_a = std::atoi(argv[4]);
  int trans_b = std::atoi(argv[5]);
  einsum_ir::model::common::Model model;

  if (std::strcmp(argv[6], "zen5") == 0) {
    model = einsum_ir::model::common::Model::ZEN5;
  } else if (std::strcmp(argv[6], "m4") == 0) {
    model = einsum_ir::model::common::Model::M4;
  } else {
    std::cerr << "Error: Unknown model '" << argv[6] << "'" << std::endl;
    std::cerr << "Available models: zen5, m4" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Matrix Config:" << std::endl;
  std::cout << "M: " << m << ", N: " << n << ", K: " << k << ", TransA: " << trans_a << ", TransB: " << trans_b << std::endl;
  std::cout << "Model: " << (model == einsum_ir::model::common::Model::ZEN5 ? "zen5" : "m4") << std::endl;

  double model_time = einsum_ir::model::common::get_time_model(m, n, k, trans_a, trans_b, model);
  double xsmm_time = einsum_ir::model::common::get_time_xsmm(m, n, k, trans_a, trans_b);

  std::cout << "Model Time: " << model_time << " seconds" << std::endl;
  std::cout << "XSMM Time: " << xsmm_time << " seconds" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  return EXIT_SUCCESS;
}