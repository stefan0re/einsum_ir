#ifndef EINSUM_IR_MODEL_ZEN5_MODEL_ZEN5_H
#define EINSUM_IR_MODEL_ZEN5_MODEL_ZEN5_H

#include <algorithm>

#include "bench_zen5.h"

namespace einsum_ir::model::zen5 {

  /**
   * TODO: Document functions
   */
  double lerp(double x0, double x1, double t);
  void find_bounds_m(const int* arr, int size, int val, int& idx_lower, double& t);
  void find_bounds_nk(const int* arr, int size, int val, int& idx_lower, double& t);
  double get_interpolated_gflops(int i_m, int i_n, int i_k, int i_trans_a, int i_trans_b);

}  // namespace einsum_ir::model::zen5

#endif  // EINSUM_IR_MODEL_ZEN5_MODEL_ZEN5_H