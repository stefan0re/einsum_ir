#ifndef EINSUM_IR_MODEL_TENSOR_TENSOR_CONSTANTS_H
#define EINSUM_IR_MODEL_TENSOR_TENSOR_CONSTANTS_H

#include "common/common.h"

namespace einsum_ir::model::tensor {

  /**
   * Hardware constants for performance modeling.
   * All bandwidth values are in GB/s.
   * All cache sizes are in bytes.
   */
  struct HardwareConstants {
    int64_t l1_size;  // L1 cache size in bytes
    int64_t l2_size;  // L2 cache size in bytes
    int64_t l3_size;  // L3 cache size in bytes

    double l1_bandwidth;   // L1 bandwidth in GB/s
    double l2_bandwidth;   // L2 bandwidth in GB/s
    double l3_bandwidth;   // L3 bandwidth in GB/s
    double mem_bandwidth;  // Main memory bandwidth in GB/s
  };

  /**
   * AMD Zen5 constants.
   */
  constexpr HardwareConstants ZEN5_CONSTANTS = {
      .l1_size = 32 * 1024,
      .l2_size = 1 * 1024 * 1024,
      .l3_size = 32 * 1024 * 1024,
      .l1_bandwidth = 500.0,
      .l2_bandwidth = 200.0,
      .l3_bandwidth = 100.0,
      .mem_bandwidth = 50.0};

  /**
   * Apple M4 constants.
   */
  constexpr HardwareConstants M4_CONSTANTS = {
      .l1_size = 128 * 1024,
      .l2_size = 16 * 1024 * 1024,
      .l3_size = 0,
      .l1_bandwidth = 912.0,
      .l2_bandwidth = 912.0,
      .l3_bandwidth = 0.0,
      .mem_bandwidth = 66.0};

  /**
   * ARM Cortex-A76 constants.
   */
  constexpr HardwareConstants A76_CONSTANTS = {
      .l1_size = 64 * 1024,
      .l2_size = 256 * 1024,
      .l3_size = 2 * 1024 * 1024,
      .l1_bandwidth = 100.0,
      .l2_bandwidth = 50.0,
      .l3_bandwidth = 30.0,
      .mem_bandwidth = 20.0};

  /**
   * Generic/default constants.
   * Conservative estimates for unknown hardware.
   */
  constexpr HardwareConstants GENERIC_CONSTANTS = {
      .l1_size = 32 * 1024,
      .l2_size = 256 * 1024,
      .l3_size = 8 * 1024 * 1024,
      .l1_bandwidth = 100.0,
      .l2_bandwidth = 50.0,
      .l3_bandwidth = 30.0,
      .mem_bandwidth = 20.0};

  /**
   * Get hardware constants for a specific architecture.
   *
   * @param arch The target architecture.
   * @return HardwareConstants for the specified architecture.
   */
  inline HardwareConstants get_hardware_constants(common::Model arch) {
    switch (arch) {
      case common::Model::ZEN5:
        return ZEN5_CONSTANTS;
      case common::Model::M4:
        return M4_CONSTANTS;
      case common::Model::A76:
        return A76_CONSTANTS;
      case common::Model::GENERIC:
      default:
        return GENERIC_CONSTANTS;
    }
  }

}  // namespace einsum_ir::model::tensor

#endif  // EINSUM_IR_MODEL_TENSOR_TENSOR_CONSTANTS_H
