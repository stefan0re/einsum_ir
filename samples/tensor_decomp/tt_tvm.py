import tvm
import tvm.auto_scheduler
import tvm_helper
import argparse
##
# Note: This script was automatically generated by tvm_from_tree.py.
#
# input_string: [[8,4],[7,3,8]->[7,3,4]],[[[[6,2,7]->[2,6,7]],[[5,1,6]->[1,5,6]]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]
##
@tvm.auto_scheduler.register_workload
def einsum_tree( dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8, dtype):
  tensor_6_2_7 = tvm.te.placeholder((dim_6, dim_2, dim_7), name='tensor_6_2_7', dtype=dtype)
  tensor_5_1_6 = tvm.te.placeholder((dim_5, dim_1, dim_6), name='tensor_5_1_6', dtype=dtype)
  tensor_8_4 = tvm.te.placeholder((dim_8, dim_4), name='tensor_8_4', dtype=dtype)
  tensor_7_3_8 = tvm.te.placeholder((dim_7, dim_3, dim_8), name='tensor_7_3_8', dtype=dtype)
  tensor_0_5 = tvm.te.placeholder((dim_0, dim_5), name='tensor_0_5', dtype=dtype)

  tmp_8 = tvm.te.reduce_axis((0, dim_8), name='tmp_8')
  tmp_7 = tvm.te.reduce_axis((0, dim_7), name='tmp_7')
  tmp_5 = tvm.te.reduce_axis((0, dim_5), name='tmp_5')
  tmp_6 = tvm.te.reduce_axis((0, dim_6), name='tmp_6')

  tensor_1_2_5_7 = tvm.te.compute( (dim_1, dim_2, dim_5, dim_7), lambda tmp_1, tmp_2, tmp_5, tmp_7: tvm.te.sum( tensor_6_2_7[ tmp_6, tmp_2, tmp_7 ] * tensor_5_1_6[ tmp_5, tmp_1, tmp_6 ] , axis=[ tmp_6 ]), name='tensor_1_2_5_7' )
  tensor_0_1_2_7 = tvm.te.compute( (dim_0, dim_1, dim_2, dim_7), lambda tmp_0, tmp_1, tmp_2, tmp_7: tvm.te.sum( tensor_1_2_5_7[ tmp_1, tmp_2, tmp_5, tmp_7 ] * tensor_0_5[ tmp_0, tmp_5 ] , axis=[ tmp_5 ]), name='tensor_0_1_2_7' )
  tensor_7_3_4 = tvm.te.compute( (dim_7, dim_3, dim_4), lambda tmp_7, tmp_3, tmp_4: tvm.te.sum( tensor_8_4[ tmp_8, tmp_4 ] * tensor_7_3_8[ tmp_7, tmp_3, tmp_8 ] , axis=[ tmp_8 ]), name='tensor_7_3_4' )
  tensor_0_1_2_3_4 = tvm.te.compute( (dim_0, dim_1, dim_2, dim_3, dim_4), lambda tmp_0, tmp_1, tmp_2, tmp_3, tmp_4: tvm.te.sum( tensor_7_3_4[ tmp_7, tmp_3, tmp_4 ] * tensor_0_1_2_7[ tmp_0, tmp_1, tmp_2, tmp_7 ] , axis=[ tmp_7 ]), name='tensor_0_1_2_3_4' )

  return [ tensor_6_2_7, tensor_5_1_6, tensor_8_4, tensor_7_3_8, tensor_0_5, tensor_0_1_2_3_4 ]

if __name__=="__main__":
  args = tvm_helper.parse_args()

  target = tvm.target.Target( tvm_helper.cpu_to_llvm( args.cpu ) )
  hardware_params = tvm.auto_scheduler.HardwareParams( target = target )
  dtype = args.dtype
  num_measure_trials = args.num_measure_trials
  timeout = args.timeout
  log_file = args.log_file

  einsum_str = "GCH,FBG,IE,HDI,AF->ABCDE"
  func = einsum_tree
  sizes = (100, 72, 128, 128, 3, 71, 305, 32, 3)

  tvm_helper.run_all( einsum_str,
                      func,
                      sizes,
                      dtype,
                      hardware_params,
                      target,
                      num_measure_trials,
                      timeout,
                      log_file )
