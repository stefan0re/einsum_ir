import etops
import numpy as np
from utils import shape_from_strides, permute_dimensions

batched_config =    etops.TensorOperationConfig(
    backend    =    "tpp",
    data_type  =    etops.float32,
    prim_first =    etops.prim.zero,
    prim_main  =    etops.prim.gemm,
    prim_last  =    etops.prim.none,
    dim_types  =   (etops.dim.c,      etops.dim.c,       etops.dim.m,     etops.dim.n,     etops.dim.k    ),
    exec_types =   (etops.exec.seq,   etops.exec.seq,    etops.exec.prim, etops.exec.prim, etops.exec.prim),
    dim_sizes  =   (32,               48,                64,              32,              128            ),
    strides    = (((128*64*48,        128*64,            1,               0,               64             ),   # in0
                   (32*128*48,        32*128,            0,               128,             1              ),   # in1
                   (32*64*48,         32*64,             1,               64,              0              )),) # out
)

top = etops.TensorOperation(batched_config)

a_shape = shape_from_strides( batched_config.dim_sizes, batched_config.strides[0][0] )
b_shape = shape_from_strides( batched_config.dim_sizes, batched_config.strides[0][1] )
c_shape = shape_from_strides( batched_config.dim_sizes, batched_config.strides[0][2] )

A = np.random.randn(*a_shape).astype(np.float32)
B = np.random.randn(*b_shape).astype(np.float32)
C = np.random.randn(*c_shape).astype(np.float32)

top.execute(A, B, C)

C_np = np.einsum("xckm,xcnk->xcnm", A, B)

# Compute absolute and relative errors
error_abs = np.max( np.abs(C - C_np) )
error_rel = np.max( np.abs(C - C_np) / (np.abs(C_np) + 1e-8) )
print("Batched config:")
print(f"  Max absolute error: {error_abs:.6e}")
print(f"  Max relative error: {error_rel:.6e}")

print(A.shape)
print(B.shape)
print(C.shape)


# Test permutation function
permuted_configs = permute_dimensions(batched_config)
print(f"Generated {len(permuted_configs)} permuted configurations")
for i, cfg in enumerate(permuted_configs):
    print(f"\nConfig {i}:")
    print(f"  dim_types:  {cfg.dim_types}")
    print(f"  exec_types: {cfg.exec_types}")
    print(f"  dim_sizes:  {cfg.dim_sizes}")
    print(f"  strides:    {cfg.strides}")