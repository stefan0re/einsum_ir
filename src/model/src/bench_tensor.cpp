#include <cstring>
#include <iostream>

#include "tensor/tensor_config.h"
#include "tensor/tensor_model.h"

using namespace einsum_ir::model::tensor;
using namespace einsum_ir::model::common;

/**
 * Parse architecture string to Model enum.
 */
Model parse_arch(const char* arch_str) {
  if (std::strcmp(arch_str, "zen5") == 0) {
    return Model::ZEN5;
  } else if (std::strcmp(arch_str, "m4") == 0) {
    return Model::M4;
  } else if (std::strcmp(arch_str, "a76") == 0) {
    return Model::A76;
  } else if (std::strcmp(arch_str, "generic") == 0) {
    return Model::GENERIC;
  }
  return Model::GENERIC;
}

/**
 * Run a simple GEMM test case.
 */
void run_simple_gemm_test(int64_t m, int64_t n, int64_t k, Model arch) {
  std::cout << "=== Simple GEMM Test ===" << std::endl;
  std::cout << "M=" << m << ", N=" << n << ", K=" << k << std::endl;

  TensorOperationConfig config = create_simple_gemm_config(m, n, k, arch);

  // Validate configuration
  ValidationResult validation = validate_config(config);
  if (!validation.valid) {
    std::cerr << "Invalid configuration: " << validation.message << std::endl;
    return;
  }

  // Print configuration
  print_config(std::cout, config);

  // Run performance model
  ModelResult result = estimate_performance(config);

  std::cout << "\n--- Model Results ---" << std::endl;
  std::cout << "Total FLOPs:    " << result.flops << std::endl;
  std::cout << "Memory bytes:   " << result.memory_bytes << std::endl;
  std::cout << "Time (seconds): " << result.time_seconds << std::endl;
  std::cout << "GFLOPS:         " << result.gflops << std::endl;
  std::cout << "Efficiency:     " << result.efficiency * 100.0 << "%" << std::endl;
  std::cout << std::endl;
}

/**
 * Run the TCCG example from the Python code.
 */
void run_tccg_example(Model arch) {
  std::cout << "=== TCCG Example ===" << std::endl;

  // Equivalent to Python:
  // dim_types  = (M, M, M, M, N, K)
  // exec_types = (SEQ, SEQ, SEQ, PRIM, PRIM, PRIM)
  // dim_sizes  = (48, 36, 36, 48, 24, 36)
  // strides    = (((2239488, 62208, 1728, 1, 0, 48),
  //                (0, 0, 0, 0, 36, 1),
  //                (1492992, 41472, 48, 1, 1728, 0)),)

  TensorOperationConfig config;
  config.primitive = prim_t::GEMM;
  config.arch = arch;

  config.dim_types = {
    dim_t::M, dim_t::M, dim_t::M, dim_t::M, dim_t::N, dim_t::K
  };

  config.exec_types = {
    exec_t::SEQ, exec_t::SEQ, exec_t::SEQ, exec_t::PRIM, exec_t::PRIM, exec_t::PRIM
  };

  config.dim_sizes = {48, 36, 36, 48, 24, 36};

  config.strides.in0 = {2239488, 62208, 1728, 1, 0, 48};
  config.strides.in1 = {0, 0, 0, 0, 36, 1};
  config.strides.out = {1492992, 41472, 48, 1, 1728, 0};

  // Validate configuration
  ValidationResult validation = validate_config(config);
  if (!validation.valid) {
    std::cerr << "Invalid configuration: " << validation.message << std::endl;
    return;
  }

  // Print configuration
  print_config(std::cout, config);

  // Run performance model
  ModelResult result = estimate_performance(config);

  std::cout << "\n--- Model Results ---" << std::endl;
  std::cout << "Total FLOPs:    " << result.flops << std::endl;
  std::cout << "Memory bytes:   " << result.memory_bytes << std::endl;
  std::cout << "Time (seconds): " << result.time_seconds << std::endl;
  std::cout << "GFLOPS:         " << result.gflops << std::endl;
  std::cout << "Efficiency:     " << result.efficiency * 100.0 << "%" << std::endl;
  std::cout << std::endl;
}

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " <command> [options]" << std::endl;
  std::cout << std::endl;
  std::cout << "Commands:" << std::endl;
  std::cout << "  simple <m> <n> <k> <arch>   Run simple GEMM test" << std::endl;
  std::cout << "  tccg <arch>                 Run TCCG example" << std::endl;
  std::cout << std::endl;
  std::cout << "Architectures: zen5, m4, a76, generic" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  " << program_name << " simple 64 48 64 zen5" << std::endl;
  std::cout << "  " << program_name << " tccg m4" << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  const char* command = argv[1];

  if (std::strcmp(command, "simple") == 0) {
    if (argc != 6) {
      std::cerr << "Error: 'simple' command requires 4 arguments" << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }

    int64_t m = std::atoll(argv[2]);
    int64_t n = std::atoll(argv[3]);
    int64_t k = std::atoll(argv[4]);
    Model arch = parse_arch(argv[5]);

    run_simple_gemm_test(m, n, k, arch);

  } else if (std::strcmp(command, "tccg") == 0) {
    if (argc != 3) {
      std::cerr << "Error: 'tccg' command requires 1 argument" << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }

    Model arch = parse_arch(argv[2]);
    run_tccg_example(arch);

  } else {
    std::cerr << "Unknown command: " << command << std::endl;
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
