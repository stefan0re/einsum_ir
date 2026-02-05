import etops
from itertools import permutations


def shape_from_strides(dim_sizes, strides):
    pairs = [(abs(s), d) for d, s in zip(dim_sizes, strides) if s != 0]
    pairs.sort(key=lambda x: x[0], reverse=True)
    return tuple(d for _, d in pairs)


def permute_dimensions(config):
    # Find indices of seq dimensions
    seq_indices = [i for i, exec_type in enumerate(config.exec_types) if exec_type == etops.exec.seq]
    
    if len(seq_indices) == 0:
        # No seq dimensions to permute
        return [config]
    
    # Get the values at seq positions
    seq_dim_types = [config.dim_types[i] for i in seq_indices]
    seq_dim_sizes = [config.dim_sizes[i] for i in seq_indices]
    
    # Generate all permutations
    configs = []
    for perm in permutations(range(len(seq_indices))):
        # Create new tuples with permuted seq values
        new_dim_types = list(config.dim_types)
        new_dim_sizes = list(config.dim_sizes)
        new_strides = []
        
        # Apply permutation to seq dimensions
        for i, seq_idx in enumerate(seq_indices):
            new_dim_types[seq_idx] = seq_dim_types[perm[i]]
            new_dim_sizes[seq_idx] = seq_dim_sizes[perm[i]]
        
        # Permute strides for each tensor
        for tensor_strides in config.strides[0]:
            seq_strides = [tensor_strides[i] for i in seq_indices]
            new_tensor_strides = list(tensor_strides)
            for i, seq_idx in enumerate(seq_indices):
                new_tensor_strides[seq_idx] = seq_strides[perm[i]]
            new_strides.append(tuple(new_tensor_strides))
        
        # Create new config
        new_config = etops.TensorOperationConfig(
            backend=config.backend,
            data_type=config.data_type,
            prim_first=config.prim_first,
            prim_main=config.prim_main,
            prim_last=config.prim_last,
            dim_types=tuple(new_dim_types),
            exec_types=config.exec_types,
            dim_sizes=tuple(new_dim_sizes),
            strides=(tuple(new_strides),)
        )
        configs.append(new_config)
    
    return configs
