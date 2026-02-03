#include "tensor_config.h"

#include <sstream>

namespace einsum_ir::model::tensor {

  ValidationResult validate_config(const TensorOperationConfig& config) {
    ValidationResult result;

    // Check primitive is valid
    if (config.primitive != prim_t::GEMM) {
      result.valid = false;
      result.message = "Only GEMM primitive is currently supported";
      return result;
    }

    // Check at least one dimension exists
    if (config.dim_types.empty()) {
      result.valid = false;
      result.message = "Configuration must have at least one dimension";
      return result;
    }

    size_t num_dims = config.dim_types.size();

    // Check all vectors have the same size
    if (config.exec_types.size() != num_dims) {
      result.valid = false;
      result.message = "exec_types size (" + std::to_string(config.exec_types.size()) +
                       ") does not match dim_types size (" + std::to_string(num_dims) + ")";
      return result;
    }

    if (config.dim_sizes.size() != num_dims) {
      result.valid = false;
      result.message = "dim_sizes size (" + std::to_string(config.dim_sizes.size()) +
                       ") does not match dim_types size (" + std::to_string(num_dims) + ")";
      return result;
    }

    // Check stride vectors match dimension count
    if (config.strides.in0.size() != num_dims) {
      result.valid = false;
      result.message = "strides.in0 size (" + std::to_string(config.strides.in0.size()) +
                       ") does not match dim_types size (" + std::to_string(num_dims) + ")";
      return result;
    }

    if (config.strides.in1.size() != num_dims) {
      result.valid = false;
      result.message = "strides.in1 size (" + std::to_string(config.strides.in1.size()) +
                       ") does not match dim_types size (" + std::to_string(num_dims) + ")";
      return result;
    }

    if (config.strides.out.size() != num_dims) {
      result.valid = false;
      result.message = "strides.out size (" + std::to_string(config.strides.out.size()) +
                       ") does not match dim_types size (" + std::to_string(num_dims) + ")";
      return result;
    }

    // Check dimension sizes are positive
    for (size_t i = 0; i < num_dims; ++i) {
      if (config.dim_sizes[i] <= 0) {
        result.valid = false;
        result.message = "dim_sizes[" + std::to_string(i) + "] must be positive, got " +
                         std::to_string(config.dim_sizes[i]);
        return result;
      }
    }

    // Check for undefined types
    for (size_t i = 0; i < num_dims; ++i) {
      if (config.dim_types[i] == dim_t::UNDEFINED_DIM) {
        result.valid = false;
        result.message = "dim_types[" + std::to_string(i) + "] is UNDEFINED_DIM";
        return result;
      }
      if (config.exec_types[i] == exec_t::UNDEFINED_EXEC) {
        result.valid = false;
        result.message = "exec_types[" + std::to_string(i) + "] is UNDEFINED_EXEC";
        return result;
      }
    }

    return result;
  }

  int64_t compute_size_product_by_dim_type(const TensorOperationConfig& config,
                                           dim_t type) {
    int64_t product = 1;
    for (size_t i = 0; i < config.dim_types.size(); ++i) {
      if (config.dim_types[i] == type) {
        product *= config.dim_sizes[i];
      }
    }
    return product;
  }

  void get_prim_mnk(const TensorOperationConfig& config,
                    int64_t& o_m,
                    int64_t& o_n,
                    int64_t& o_k) {
    o_m = 1;
    o_n = 1;
    o_k = 1;

    int id_m = -1;
    int id_n = -1;
    int id_k = -1;

    for (size_t i = 0; i < config.dim_types.size(); ++i) {
      if (config.exec_types[i] == exec_t::PRIM) {
        switch (config.dim_types[i]) {
          case dim_t::M:
            o_m *= config.dim_sizes[i];
            id_m = i;
            break;
          case dim_t::N:
            o_n *= config.dim_sizes[i];
            id_n = i;
            break;
          case dim_t::K:
            o_k *= config.dim_sizes[i];
            id_k = i;
            break;
          default:
            break;
        }
      }
    }

    // check for unit stride dimension
    if (config.strides.in0[id_m] != 1 && config.strides.in0[id_k] != 1) {
      o_m = 1;
      o_k = 1;
    }
    if (config.strides.in1[id_n] != 1 && config.strides.in1[id_k] != 1) {
      o_n = 1;
      o_k = 1;
    }
    if (config.strides.out[id_m] != 1) {
      o_m = 1;
      o_n = 1;
    }
  }

  void get_gemm_transpose_flags(const TensorOperationConfig& config,
                                int32_t& o_trans_a,
                                int32_t& o_trans_b) {
    o_trans_a = 0;
    o_trans_b = 0;

    for (size_t i = 0; i < config.dim_types.size(); ++i) {
      if (config.exec_types[i] == exec_t::PRIM) {
        if (config.dim_types[i] == dim_t::K) {
          if (config.strides.in0[i] == 1) {
            o_trans_a = 1;
          }
        }
        if (config.dim_types[i] == dim_t::N) {
          if (config.strides.in1[i] == 1) {
            o_trans_b = 1;
          }
        }
      }
    }
  }

  int64_t get_loop_iterations(const TensorOperationConfig& config) {
    int64_t iterations = 1;
    for (size_t i = 0; i < config.exec_types.size(); ++i) {
      if (config.exec_types[i] == exec_t::SEQ ||
          config.exec_types[i] == exec_t::SHARED) {
        iterations *= config.dim_sizes[i];
      }
    }
    return iterations;
  }

  double get_total_tensor_sizes(const TensorOperationConfig& config) {
    double total_size = 0.0;

    // Input tensor 0
    double in0_size = dtype_size(config.dtype_in);
    double in1_size = dtype_size(config.dtype_in);
    double out_size = dtype_size(config.dtype_out);
    for (size_t i = 0; i < config.dim_sizes.size(); ++i) {
      if (config.strides.in0[i] != 0)
        in0_size *= config.dim_sizes[i];
      if (config.strides.in1[i] != 0)
        in1_size *= config.dim_sizes[i];
      if (config.strides.out[i] != 0)
        out_size *= config.dim_sizes[i];
    }
    total_size += in0_size + in1_size + out_size;

    return total_size;
  }

  void print_config(std::ostream& os, const TensorOperationConfig& config) {
    os << "TensorOperationConfig {\n";
    os << "  primitive: " << prim_to_string(config.primitive) << "\n";

    os << "  dim_types:  [";
    for (size_t i = 0; i < config.dim_types.size(); ++i) {
      os << dim_to_string(config.dim_types[i]);
      if (i < config.dim_types.size() - 1) os << ", ";
    }
    os << "]\n";

    os << "  exec_types: [";
    for (size_t i = 0; i < config.exec_types.size(); ++i) {
      os << exec_to_string(config.exec_types[i]);
      if (i < config.exec_types.size() - 1) os << ", ";
    }
    os << "]\n";

    os << "  dim_sizes:  [";
    for (size_t i = 0; i < config.dim_sizes.size(); ++i) {
      os << config.dim_sizes[i];
      if (i < config.dim_sizes.size() - 1) os << ", ";
    }
    os << "]\n";

    os << "  strides.in0: [";
    for (size_t i = 0; i < config.strides.in0.size(); ++i) {
      os << config.strides.in0[i];
      if (i < config.strides.in0.size() - 1) os << ", ";
    }
    os << "]\n";

    os << "  strides.in1: [";
    for (size_t i = 0; i < config.strides.in1.size(); ++i) {
      os << config.strides.in1[i];
      if (i < config.strides.in1.size() - 1) os << ", ";
    }
    os << "]\n";

    os << "  strides.out: [";
    for (size_t i = 0; i < config.strides.out.size(); ++i) {
      os << config.strides.out[i];
      if (i < config.strides.out.size() - 1) os << ", ";
    }
    os << "]\n";

    os << "  arch: ";
    switch (config.arch) {
      case common::Model::ZEN5:
        os << "ZEN5";
        break;
      case common::Model::M4:
        os << "M4";
        break;
      case common::Model::A76:
        os << "A76";
        break;
      case common::Model::GENERIC:
        os << "GENERIC";
        break;
    }
    os << "\n";

    os << "  dtype_in0:  " << dtype_to_string(config.dtype_in) << "\n";
    os << "  dtype_in1:  " << dtype_to_string(config.dtype_in) << "\n";
    os << "  dtype_out:  " << dtype_to_string(config.dtype_out) << "\n";
    os << "  dtype_comp: " << dtype_to_string(config.dtype_comp) << "\n";

    // Print derived info
    int64_t prim_m, prim_n, prim_k;
    get_prim_mnk(config, prim_m, prim_n, prim_k);
    os << "  -- derived --\n";
    os << "  prim_m: " << prim_m << ", prim_n: " << prim_n << ", prim_k: " << prim_k << "\n";
    os << "  loop_iterations: " << get_loop_iterations(config) << "\n";
    os << "}\n";
  }

  TensorOperationConfig create_simple_gemm_config(int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  common::Model arch) {
    TensorOperationConfig config;
    config.primitive = prim_t::GEMM;
    config.arch = arch;

    // Simple GEMM: 3 dimensions, all handled by primitive
    config.dim_types = {dim_t::M, dim_t::N, dim_t::K};
    config.exec_types = {exec_t::PRIM, exec_t::PRIM, exec_t::PRIM};
    config.dim_sizes = {m, n, k};

    // Column-major strides for A (m x k), B (k x n), C (m x n)
    // A: stride_m = 1, stride_n = 0, stride_k = m
    // B: stride_m = 0, stride_n = k, stride_k = 1
    // C: stride_m = 1, stride_n = m, stride_k = 0
    config.strides.in0 = {1, 0, m};  // A[m,k]
    config.strides.in1 = {0, k, 1};  // B[k,n]
    config.strides.out = {1, m, 0};  // C[m,n]

    return config;
  }

}  // namespace einsum_ir::model::tensor
