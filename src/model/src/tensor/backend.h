#ifndef EINSUM_IR_MODEL_TENSOR_BACKEND_H
#define EINSUM_IR_MODEL_TENSOR_BACKEND_H

#include "../common/common.h"
#include "config.h"
#include "frontend.h"

namespace einsum_ir::model::tensor::backend {

  /**
   * Calculates theoretical compute time using performance model.
   *
   * @param config Tensor operation configuration
   * @param o_gflops Output parameter for achieved GFLOPS
   * @return Theoretical compute time in seconds
   */
  double compute_time(frontend::TensorOperationConfig const& config,
                      double& o_gflops);

  /**
   * Calculates theoretical memory transfer time.
   *
   * @param config Tensor operation configuration
   * @param o_gflops Output parameter for achieved GFLOPS
   * @return Theoretical memory time in seconds
   */
  double memory_time(frontend::TensorOperationConfig const& config,
                     double& o_gflops);

}  // namespace einsum_ir::model::tensor::backend

#endif  // EINSUM_IR_MODEL_TENSOR_BACKEND_H
