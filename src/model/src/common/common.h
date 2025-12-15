#ifndef EINSUM_IR_MODEL_COMMON_COMMON_H
#define EINSUM_IR_MODEL_COMMON_COMMON_H

namespace einsum_ir::model::common {

  /**
   * Enum representing available performance models.
   */
  enum class Model {
    ZEN5,
    M4,
    PI5,
    GENERIC
  };

  /**
   * Get the estimated execution time using a performance model.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_trans_a The transpose flag for matrix A (0 or 1).
   * @param i_trans_b The transpose flag for matrix B (0 or 1).
   * @param i_model The performance model to use.
   *
   * @return The estimated execution time in seconds.
   */
  double get_time_model(int i_m,
                        int i_n,
                        int i_k,
                        int i_trans_a,
                        int i_trans_b,
                        Model i_model);

  /**
   * Get the measured execution time using libxsmm.
   *
   * @param i_m The M dimension size.
   * @param i_n The N dimension size.
   * @param i_k The K dimension size.
   * @param i_trans_a The transpose flag for matrix A (0 or 1).
   * @param i_trans_b The transpose flag for matrix B (0 or 1).
   *
   * @return The measured execution time in seconds.
   */
  double get_time_xsmm(int i_m,
                       int i_n,
                       int i_k,
                       int i_trans_a,
                       int i_trans_b);

}  // namespace einsum_ir::model::common

#endif  // EINSUM_IR_MODEL_COMMON_COMMON_H
