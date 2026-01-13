import etops

top_config = etops.TensorOperationConfig(
    backend    =   "tpp",
    data_type  =   etops.float32,
    prim_first =   etops.prim.zero,
    prim_main  =   etops.prim.gemm,
    prim_last  =   etops.prim.none,
    dim_types  =   (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
    exec_types =   (etops.exec.prim, etops.exec.prim, etops.exec.prim),
    dim_sizes  =   (512,             512,             512            ),
    strides    = (((1,               0,               512            ),   # in0
                   (0,               512,             1              ),   # in1
                   (1,               512,             0              )),) # out
)

# Create the TensorOperation instance
top = etops.TensorOperation(top_config)

# Create input and output arrays
import numpy as np
A = np.random.randn(512,512).astype(np.float32)
B = np.random.randn(512,512).astype(np.float32)
C = np.random.randn(512,512).astype(np.float32)

# Execute the operation
top.execute(A, B, C)

C_np = np.einsum("km,nk->nm", A, B)

# Compute absolute and relative errors
error_abs = np.max( np.abs(C - C_np) )
error_rel = np.max( np.abs(C - C_np) / (np.abs(C_np) + 1e-8) )
print("Column-major GEMM operation:")
print(f"  Max absolute error: {error_abs:.6e}")
print(f"  Max relative error: {error_rel:.6e}")

# benchmarking 
import time

num_iters = 100000

start_time = time.time()
for _ in range(num_iters):
    top.execute(A, B, C)
end_time = time.time()

total_time = end_time - start_time

avg_time_per_iter = total_time / num_iters

print(f"Execution time: {avg_time_per_iter:.6f} s")


gflops = (2.0 * 512 * 512 * 512) / (avg_time_per_iter * 1e9)
print(f"Achieved GFLOPS: {gflops:.2f} GFLOPS")

estimated_time = top.model_gemm(etops.ModelType.m4)
print(f"Estimated time by model: {estimated_time:.20f} s")

es_gflops = (2.0 * 512 * 512 * 512) / (estimated_time * 1e9)
print(f"Estimated GFLOPS by model: {es_gflops:.2f} GFLOPS")