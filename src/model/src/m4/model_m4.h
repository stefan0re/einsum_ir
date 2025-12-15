#ifndef EINSUM_IR_MODEL_M4_MODEL_M4_H
#define EINSUM_IR_MODEL_M4_MODEL_M4_H

#include <algorithm>

#include "bench_m4.h"

namespace einsum_ir::model::m4 {

  /**
   * Linear interpolation function.
   * @param x0 The value at t=0.
   * @param x1 The value at t=1.
   * @param t The interpolation factor between 0 and 1.
   *
   * @return The interpolated value.
   */
  double lerp(double x0, double x1, double t);

  /**
   * Find surrounding indices and interpolation factor for M and N dimensions
   *
   * @param arr The array of dimension values.
   * @param size The size of the dimension array.
   * @param val The target dimension value.
   * @param idx_lower Output parameter for the lower index.
   * @param t Output parameter for the interpolation factor.
   */
  void find_bounds_mn(const int* arr,
                      int size,
                      int val,
                      int& idx_lower,
                      double& t);
  /**
   * Find surrounding indices and interpolation factor for K dimension
   *
   * @param arr The array of dimension values.
   * @param size The size of the dimension array.
   * @param val The target dimension value.
   * @param idx_lower Output parameter for the lower index.
   * @param t Output parameter for the interpolation factor.
   */
  void find_bounds_k(const int* arr,
                     int size,
                     int val,
                     int& idx_lower,
                     double& t);

  /**
   * Get interpolated GFLOPS value based on input dimensions and transpose flag.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_trans_b The transpose flag for matrix B (0 or 1).
   *
   * @return The interpolated GFLOPS value.
   */
  double get_interpolated_gflops(int i_m, int i_n, int i_k, int i_trans_b);

}  // namespace einsum_ir::model::m4

#endif  // EINSUM_IR_MODEL_M4_MODEL_M4_H