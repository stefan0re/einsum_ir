#include <cstring>
#include <iostream>

#include "common/common.h"

int main(int argc, char** argv) {
  if (argc < 7 || argc > 9) {
    std::cout << "Usage: " << argv[0] << " <m> <n> <k> <trans_a> <trans_b> <model> [peak_gflops] [vector_size]" << std::endl;
    std::cout << "  m, n, k:       Matrix dimensions (positive integers)" << std::endl;
    std::cout << "  trans_a:       Transpose A matrix (0 or 1)" << std::endl;
    std::cout << "  trans_b:       Transpose B matrix (0 or 1)" << std::endl;
    std::cout << "  model:         Performance model to use (zen5, m4, a76, generic)" << std::endl;
    std::cout << "  peak_gflops:   [Optional, generic only] Peak GFLOPS of architecture" << std::endl;
    std::cout << "  vector_size:   [Optional, generic only] Vector width in byte" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << argv[0] << " 64 48 64 0 0 zen5" << std::endl;
    std::cout << "  " << argv[0] << " 64 48 64 0 0 generic 100.0 8" << std::endl;
    return EXIT_FAILURE;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);
  int trans_a = std::atoi(argv[4]);
  int trans_b = std::atoi(argv[5]);
  einsum_ir::model::common::Model model;

  // Optional parameters for generic model
  double peak_gflops = 0.0;
  int vector_size = 0;

  if (std::strcmp(argv[6], "zen5") == 0) {
    model = einsum_ir::model::common::Model::ZEN5;
  } else if (std::strcmp(argv[6], "m4") == 0) {
    model = einsum_ir::model::common::Model::M4;
  } else if (std::strcmp(argv[6], "a76") == 0) {
    model = einsum_ir::model::common::Model::A76;
  } else if (std::strcmp(argv[6], "generic") == 0) {
    model = einsum_ir::model::common::Model::GENERIC;

    // Parse optional generic parameters
    if (argc >= 8) {
      peak_gflops = std::atof(argv[7]);
    }
    if (argc >= 9) {
      vector_size = std::atoi(argv[8]);
    }

    // Validate generic parameters
    if (peak_gflops <= 0.0 || vector_size <= 0) {
      std::cerr << "Error: For generic model, you must provide peak_gflops > 0 and vector_size > 0" << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    std::cerr << "Error: Unknown model '" << argv[6] << "'" << std::endl;
    std::cerr << "Available models: zen5, m4, a76, generic" << std::endl;
    return EXIT_FAILURE;
  }

  // Warn if extra parameters provided for non-generic models
  if (model != einsum_ir::model::common::Model::GENERIC && argc > 7) {
    std::cerr << "Warning: Extra parameters ignored for non-generic models" << std::endl;
  }

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Matrix Config:" << std::endl;
  std::cout << "M: " << m << ", N: " << n << ", K: " << k << ", TransA: " << trans_a << ", TransB: " << trans_b << std::endl;
  std::cout << "Model: " << (model == einsum_ir::model::common::Model::ZEN5 ? "zen5" : model == einsum_ir::model::common::Model::M4 ? "m4"
                                                                                   : model == einsum_ir::model::common::Model::A76  ? "a76"
                                                                                                                                    : "generic")
            << std::endl;

  if (model == einsum_ir::model::common::Model::GENERIC) {
    std::cout << "Peak GFLOPS: " << peak_gflops << std::endl;
    std::cout << "Vector Size: " << vector_size << std::endl;
  }

  double model_time = einsum_ir::model::common::get_time_model(m, n, k, trans_a, trans_b, model, peak_gflops, vector_size);
  double xsmm_time = einsum_ir::model::common::get_time_xsmm(m, n, k, trans_a, trans_b);

  std::cout << "Model Time: " << model_time << " seconds" << std::endl;
  std::cout << "XSMM Time: " << xsmm_time << " seconds" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  return EXIT_SUCCESS;
}