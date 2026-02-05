from tccg_configs import tccg_configs
import etops
import numpy as np
from utils import shape_from_strides, permute_dimensions
import time

# Then iterate through all configs
for i, config in enumerate(tccg_configs):
    print(f"\nExecuting config {i+1}...")
    
    # Create tensor operation
    top = etops.TensorOperation(config)
    
    # Create input tensors based on strides
    a_shape = shape_from_strides(config.dim_sizes, config.strides[0][0])
    b_shape = shape_from_strides(config.dim_sizes, config.strides[0][1])
    c_shape = shape_from_strides(config.dim_sizes, config.strides[0][2])
    
    A = np.random.randn(*a_shape).astype(np.float32)
    B = np.random.randn(*b_shape).astype(np.float32)
    C = np.random.randn(*c_shape).astype(np.float32)
    
    # benchmarking
    top.execute(A, B, C)

    iter = 100
    t0 = time.perf_counter()
    for _ in range(iter):
        top.execute(A, B, C)
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / iter
    print(f"Average execution time: {elapsed:.4f} s")
    print(f'Estimated time:         {top.model(etops.ModelType.m4):.4f} s')
    print(f"Diff:                   {abs(elapsed - top.model(etops.ModelType.m4)):.4f} s")
