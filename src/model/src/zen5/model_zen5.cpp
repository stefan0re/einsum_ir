#include "model_zen5.h"

namespace einsum_ir::model::zen5 {

  double lerp(double x0, double x1, double t) {
    return x0 + t * (x1 - x0);
  }

  void find_bounds_m(const int* arr, int size, int val, int& idx_lower, double& t) {
    // Check if value is above 128
    if (val > 128) {
      // Map to 113-128 range based on modulo 16
      // val % 16 gives 0-15, map to values 113-128
      // 113 % 16 = 1, 114 % 16 = 2, ..., 127 % 16 = 15, 128 % 16 = 0
      int mod16 = val % 16;
      int mapped_val = (mod16 == 0) ? 128 : (112 + mod16);

      // Now find this mapped value in the array
      const int* exact = std::lower_bound(arr, arr + size, mapped_val);
      if (exact != arr + size && *exact == mapped_val) {
        idx_lower = exact - arr;
        t = 0.0;
        return;
      }
    }

    // Exact match check
    const int* exact = std::lower_bound(arr, arr + size, val);
    if (exact != arr + size && *exact == val) {
      idx_lower = exact - arr;
      t = 0.0;
      return;
    }

    // Find surrounding values
    const int* upper = std::upper_bound(arr, arr + size, val);

    // Handle out of range with clamping
    if (upper == arr) {
      // Below minimum - clamp to first value
      idx_lower = 0;
      t = 0.0;
      return;
    }
    if (upper == arr + size) {
      // Above maximum (shouldn't reach here due to above check)
      idx_lower = size - 1;
      t = 0.0;
      return;
    }

    // Interpolate between lower and upper
    const int* lower = upper - 1;
    idx_lower = lower - arr;

    double v_lower = *lower;
    double v_upper = *upper;
    t = (val - v_lower) / (v_upper - v_lower);
  }

  void find_bounds_nk(const int* arr, int size, int val, int& idx_lower, double& t) {
    // Exact match check
    const int* exact = std::lower_bound(arr, arr + size, val);
    if (exact != arr + size && *exact == val) {
      idx_lower = exact - arr;
      t = 0.0;
      return;
    }

    // Find surrounding values
    const int* upper = std::upper_bound(arr, arr + size, val);

    // Handle out of range with clamping
    if (upper == arr) {
      // Below minimum - clamp to first value
      idx_lower = 0;
      t = 0.0;
      return;
    }
    if (upper == arr + size) {
      // Above maximum - clamp to last value (use as exact match)
      idx_lower = size - 1;
      t = 0.0;
      return;
    }

    // Interpolate between lower and upper
    const int* lower = upper - 1;
    idx_lower = lower - arr;

    double v_lower = *lower;
    double v_upper = *upper;
    t = (val - v_lower) / (v_upper - v_lower);
  }

  double get_interpolated_gflops(int i_m, int i_n, int i_k, int i_trans_a, int i_trans_b) {
    // Clamp trans_a and trans_b to valid range
    if (i_trans_a < 0) i_trans_a = 0;
    if (i_trans_a > 1) i_trans_a = 1;
    if (i_trans_b < 0) i_trans_b = 0;
    if (i_trans_b > 1) i_trans_b = 1;

    // Find surrounding indices and interpolation factors
    int m_idx0, n_idx0, k_idx0;
    double t_m, t_n, t_k;

    find_bounds_m(M_VALUES, M_SIZE, i_m, m_idx0, t_m);
    find_bounds_nk(N_VALUES, N_SIZE, i_n, n_idx0, t_n);
    find_bounds_nk(K_VALUES, K_SIZE, i_k, k_idx0, t_k);

    // For clamped values (t=0 and idx at boundary), keep same index
    int m_idx1 = (t_m > 0.0 && m_idx0 + 1 < M_SIZE) ? m_idx0 + 1 : m_idx0;
    int n_idx1 = (t_n > 0.0 && n_idx0 + 1 < N_SIZE) ? n_idx0 + 1 : n_idx0;
    int k_idx1 = (t_k > 0.0 && k_idx0 + 1 < K_SIZE) ? k_idx0 + 1 : k_idx0;

    // Get 8 corner values for trilinear interpolation
    double c000 = gflops_table[m_idx0][n_idx0][k_idx0][i_trans_a][i_trans_b];
    double c100 = gflops_table[m_idx1][n_idx0][k_idx0][i_trans_a][i_trans_b];
    double c010 = gflops_table[m_idx0][n_idx1][k_idx0][i_trans_a][i_trans_b];
    double c110 = gflops_table[m_idx1][n_idx1][k_idx0][i_trans_a][i_trans_b];
    double c001 = gflops_table[m_idx0][n_idx0][k_idx1][i_trans_a][i_trans_b];
    double c101 = gflops_table[m_idx1][n_idx0][k_idx1][i_trans_a][i_trans_b];
    double c011 = gflops_table[m_idx0][n_idx1][k_idx1][i_trans_a][i_trans_b];
    double c111 = gflops_table[m_idx1][n_idx1][k_idx1][i_trans_a][i_trans_b];

    // Trilinear interpolation
    // First interpolate in m direction (4 values)
    double c00 = lerp(c000, c100, t_m);
    double c01 = lerp(c001, c101, t_m);
    double c10 = lerp(c010, c110, t_m);
    double c11 = lerp(c011, c111, t_m);

    // Then interpolate in n direction (2 values)
    double c0 = lerp(c00, c10, t_n);
    double c1 = lerp(c01, c11, t_n);

    // Finally interpolate in k direction (1 value)
    double result = lerp(c0, c1, t_k);

    return result;
  }

}  // namespace einsum_ir::model::zen5
