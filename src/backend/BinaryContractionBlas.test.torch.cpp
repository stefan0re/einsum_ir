#include "ATen/ATen.h"
#include "catch.hpp"
#include "BinaryContractionBlas.h"

TEST_CASE( "FP32 BLAS-based binary contraction executing a matmul.", "[binary_contraction_blas]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );

  int64_t l_dim_ids_in_left[2]  = { 2, 0 };
  int64_t l_dim_ids_in_right[2] = { 1, 2 };
  int64_t l_dim_ids_out[2]      = { 1, 0 };

  // data layout
  //
  //    ____nm___
  //   /         \
  // km           nk
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 2,
                   2,
                   2,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   nullptr,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_in_left  = at::rand( {4, 2} );
  at::Tensor l_in_right = at::rand( {3, 4} );
  at::Tensor l_out_ref  = at::rand( {3, 2} );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref += at::einsum( "km,nk->nm",
                           {l_in_left, l_in_right} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  l_bin_cont.contract( l_in_left.data_ptr(),
                       l_in_right.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_ref, l_out_native )  );
}

TEST_CASE( "FP64 BLAS-based binary contraction executing a packed matmul.", "[binary_contraction_blas]" ) {
  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 3 ) );

  int64_t l_dim_ids_in_left[3]  = { 2, 0, 3 };
  int64_t l_dim_ids_in_right[3] = { 1, 2, 3 };
  int64_t l_dim_ids_out[3]      = { 1, 0, 3 };

  // data layout
  //
  //    ____nmc___
  //   /          \
  // kmc           nkc
  //
  // char   id   size
  //    m    0      2
  //    n    1      3
  //    k    2      4
  //    c    3      3
  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 3,
                   3,
                   3,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   nullptr,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::FP64,
                   einsum_ir::UNDEFINED_KTYPE,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  // data
  at::Tensor l_left    = at::rand( {4, 2, 3},
                                   at::dtype( at::kDouble ) );
  at::Tensor l_right   = at::rand( {3, 4, 3},
                                   at::dtype( at::kDouble ) );
  at::Tensor l_out_ref = at::rand( {3, 2, 3},
                                   at::dtype( at::kDouble ) );
  at::Tensor l_out_native = l_out_ref.clone();

  // reference
  l_out_ref += at::einsum( "kmc,nkc->nmc",
                           {l_left, l_right} );

  // compile contraction
  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // execute
  at::Tensor l_left_ordered  = l_left.permute( { 2, 0, 1 } ).contiguous();
  at::Tensor l_right_ordered = l_right.permute( { 2, 0, 1 } ).contiguous();
  l_bin_cont.contract( l_left_ordered.data_ptr(),
                       l_right_ordered.data_ptr(),
                       l_out_native.data_ptr() );

  REQUIRE( at::allclose( l_out_ref, l_out_native )  );
}

TEST_CASE( "FP32 BLAS-based binary contraction involving C, M, N and K dimensions, stride-1 M.", "[binary_contraction_blas]" ) {
  // Test case:
  //
  //         ______________yhgfxei________________
  //        /                                     \
  //   ygcxaei                                   yhcxfa
  //
  //   char id size type
  //      i  0    3   m0
  //      e  1    8   m1
  //      a  2    2   k0
  //      c  3    7   k1
  //      g  4    6   m2
  //      f  5    5   n0
  //      h  6    4   n1
  //      x  7    3   c0
  //      y  8    4   c1
  //
  //  yhgfxei: 8 4 3 7 2 1 0
  //  yhcxfa:  8 6 3 7 5 2
  //  ygcxaei: 8 6 4 5 7 1 0
  //
  //   dim types:
  //     c:  yx /  87
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 8 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 7 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4, 6 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 5, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 6, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 7, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 8, 4 ) );

  int64_t l_dim_ids_in_left[7] = { 8, 4, 3, 7, 2, 1, 0 };
  int64_t l_dim_ids_in_right[6] = { 8, 6, 3, 7, 5, 2 };
  int64_t l_dim_ids_out[7] = { 8, 6, 4, 5, 7, 1, 0 };

  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 7,
                   6,
                   7,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   nullptr,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  //                                0  1  2  3  4  5  6
  //                                y  g  c  x  a  e  i
  at::Tensor l_in_left = at::randn( {4, 6, 7, 3, 2, 8, 3} );
  //                                 0  1  2  3  4  5
  //                                 y  h  c  x  f  a
  at::Tensor l_in_right = at::randn( {4, 4, 7, 3, 5, 2} );
  //                                y  h  g  f  x  e  i
  at::Tensor l_out_ref = at::rand( {4, 4, 6, 5, 3, 8, 3} );
  at::Tensor l_out_native = l_out_ref.clone();
  at::Tensor l_out_ordered = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "ygcxaei,yhcxfa->yhgfxei",
                          {l_in_left, l_in_right} );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // BLAS call will use blocking:
  //   mb: e, i
  //   nb: f
  //   kb: c, a
  // ordering:
  //   left  (BC-BM-BK-KB-MB): yx - g - - ca - ei
  //   right (BC-BN-BK-NB-KB): yx - h - - f  - ca
  at::Tensor l_left_ordered  = l_in_left.permute(  { 0, 3, 1, 2, 4, 5, 6 } ).contiguous();
  at::Tensor l_right_ordered = l_in_right.permute( { 0, 3, 1, 4, 2, 5 } ).contiguous();

  l_bin_cont.contract( l_left_ordered.data_ptr(),
                       l_right_ordered.data_ptr(),
                       l_out_ordered.data_ptr() );

  REQUIRE( at::allclose( l_out_ordered, l_out_ref, 1E-4, 1E-6 )  );
}

TEST_CASE( "FP32 BLAS-based binary contraction involving C, M, N and K dimensions, stride-1 C.", "[binary_contraction_blas]" ) {
  // Test case:
  //
  //         ______________hgfxeiy________________
  //        /                                     \
  //   ygcxaei                                   yhcxfa
  //
  //   char id size type
  //      i  0    3   m0
  //      e  1    8   m1
  //      a  2    2   k0
  //      c  3    7   k1
  //      g  4    6   m2
  //      f  5    5   n0
  //      h  6    4   n1
  //      x  7    3   c0
  //      y  8    4   c1
  //
  //  yhgfxei: 4 3 7 2 1 0 8
  //  yhcxfa:  8 6 3 7 5 2
  //  ygcxaei: 8 6 4 5 7 1 0
  //
  //   dim types:
  //     c:  yx /  87
  //     m: gei / 410
  //     n:  hf /  65
  //     k:  ca /  32

  std::map< int64_t, int64_t > l_dim_sizes;
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 0, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 1, 8 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 2, 2 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 3, 7 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 4, 6 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 5, 5 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 6, 4 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 7, 3 ) );
  l_dim_sizes.insert( std::pair< int64_t, int64_t >( 8, 4 ) );

  int64_t l_dim_ids_in_left[7] = { 8, 4, 3, 7, 2, 1, 0 };
  int64_t l_dim_ids_in_right[6] = { 8, 6, 3, 7, 5, 2 };
  int64_t l_dim_ids_out[7] = { 6, 4, 5, 7, 1, 0, 8 };

  einsum_ir::backend::BinaryContractionBlas l_bin_cont;
  l_bin_cont.init( 7,
                   6,
                   7,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   &l_dim_sizes,
                   nullptr,
                   &l_dim_sizes,
                   nullptr,
                   nullptr,
                   nullptr,
                   l_dim_ids_in_left,
                   l_dim_ids_in_right,
                   l_dim_ids_out,
                   nullptr,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::FP32,
                   einsum_ir::ZERO,
                   einsum_ir::MADD,
                   einsum_ir::UNDEFINED_KTYPE );

  //                              0  1  2  3  4  5  6
  //                              y  g  c  x  a  e  i
  at::Tensor l_left = at::randn( {4, 6, 7, 3, 2, 8, 3} );
  //                               0  1  2  3  4  5
  //                               y  h  c  x  f  a
  at::Tensor l_right = at::randn( {4, 4, 7, 3, 5, 2} );
  //                                h  g  f  x  e  i  y
  at::Tensor l_out_ref = at::rand( {4, 6, 5, 3, 8, 3, 4} );
  at::Tensor l_out_native = l_out_ref.clone();
  at::Tensor l_out_ordered = l_out_ref.clone();

  // reference
  l_out_ref = at::einsum( "ygcxaei,yhcxfa->hgfxeiy",
                          {l_left, l_right} );

  einsum_ir::err_t l_err = l_bin_cont.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  // BLAS call will use blocking:
  //   cb: y
  //   mb: e, i
  //   nb: f
  //   kb: c, a
  // ordering:
  //   left  (BC-BM-BK-CB-KB-MB): x - g - - y - ca - ei
  //   right (BC-BN-BK-CB-NB-KB): x - h - - y - f  - ca
  at::Tensor l_left_ordered  = l_left.permute(  { 3, 1, 0, 2, 4, 5, 6 } ).contiguous();
  at::Tensor l_right_ordered = l_right.permute( { 3, 1, 0, 4, 2, 5 } ).contiguous();

  l_bin_cont.contract( l_left_ordered.data_ptr(),
                       l_right_ordered.data_ptr(),
                       l_out_ordered.data_ptr() );

  REQUIRE( at::allclose( l_out_ordered, l_out_ref, 1E-4, 1E-6 )  );
}