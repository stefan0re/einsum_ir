#ifndef EINSUM_IR_MODEL_TENSOR_CONFIG_H
#define EINSUM_IR_MODEL_TENSOR_CONFIG_H

#include <cstdint>

namespace einsum_ir::model::tensor {

  // Hardware bandwidth constants (in GB/s)
  static constexpr double L1_BANDWIDTH_GB_S = 182.0;
  static constexpr double L2_BANDWIDTH_GB_S = 160.0;
  static constexpr double L3_BANDWIDTH_GB_S = 60.0;
  static constexpr double DRAM_BANDWIDTH_GB_S = 29.0;

}  // namespace einsum_ir::model::tensor

#endif  // EINSUM_IR_MODEL_TENSOR_CONFIG_H
