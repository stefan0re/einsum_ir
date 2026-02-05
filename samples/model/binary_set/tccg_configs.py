

import etops

tccg_configs = []


tccg_01 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (48, 36, 36, 48, 24, 36 ),
    strides    = (((2239488, 62208, 1728, 1, 0, 48 ),
                   (0, 0, 0, 0, 36, 1 ),
                   (1492992, 41472, 48, 1, 1728, 0 )),)
)

tccg_configs.append(tccg_01)


tccg_02 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (48, 36, 36, 48, 24, 36 ),
    strides    = (((2239488, 62208, 1728, 1, 0, 48 ),
                   (0, 0, 0, 0, 36, 1 ),
                   (1492992, 1728, 48, 1, 62208, 0 )),)
)

tccg_configs.append(tccg_02)

tccg_03 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (96, 84, 96, 24, 96 ),
    strides    = (((774144, 9216, 1, 0, 96 ),
                   (0, 0, 0, 96, 1 ),
                   (193536, 2304, 1, 96, 0 )),)
)

tccg_configs.append(tccg_03)

tccg_04 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (48, 36, 36, 48, 24, 48 ),
    strides    = (((2985984, 82944, 2304, 1, 0, 48 ),
                   (0, 0, 0, 0, 48, 1 ),
                   (1492992, 41472, 1152, 1, 48, 0 )),)
)

tccg_configs.append(tccg_04)


tccg_05 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (96, 84, 96, 24, 84 ),
    strides    = (((677376, 8064, 1, 0, 96 ),
                   (0, 0, 0, 84, 1 ),
                   (193536, 96, 1, 8064, 0 )),)
)

tccg_configs.append(tccg_05)

tccg_06 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.k, etops.dim.n, etops.dim.n, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (6, 6, 64, 6, 24, 64, 64 ),
    strides    = (((1536, 0, 0, 0, 1, 0, 24 ),
                   (64, 9437184, 147456, 24576, 0, 384, 1 ),
                   (0, 589824, 9216, 1536, 1, 24, 0 )),)
)

tccg_configs.append(tccg_06)

tccg_07 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (96, 84, 84, 24, 96 ),
    strides    = (((677376, 8064, 1, 0, 84 ),
                   (0, 0, 0, 96, 1 ),
                   (169344, 2016, 1, 84, 0 )),)
)

tccg_configs.append(tccg_07)

tccg_08 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (20, 24, 20, 24, 20, 20, 24 ),
    strides    = (((9600, 0, 480, 0, 1, 0, 20 ),
                   (0, 11520, 0, 480, 0, 24, 1 ),
                   (192000, 3840000, 9600, 400, 1, 20, 0 )),)
)

tccg_configs.append(tccg_08)

tccg_09 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (20, 24, 24, 20, 20, 20, 24 ),
    strides    = (((11520, 0, 480, 0, 1, 0, 20 ),
                   (0, 9600, 0, 480, 0, 24, 1 ),
                   (192000, 3840000, 400, 9600, 1, 20, 0 )),)
)

tccg_configs.append(tccg_09)

tccg_10 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (24, 20, 20, 24, 20, 20, 24 ),
    strides    = (((9600, 0, 480, 0, 1, 0, 20 ),
                   (0, 11520, 0, 480, 0, 24, 1 ),
                   (3840000, 192000, 9600, 400, 1, 20, 0 )),)
)

tccg_configs.append(tccg_10)

tccg_11 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (24, 20, 20, 24, 20, 20, 24 ),
    strides    = (((9600, 0, 480, 0, 1, 0, 20 ),
                   (0, 11520, 0, 480, 0, 24, 1 ),
                   (3840000, 9600, 192000, 400, 1, 20, 0 )),)
)

tccg_configs.append(tccg_11)

tccg_12 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (6, 6, 64, 4, 94, 24, 64 ),
    strides    = (((9240576, 6016, 144384, 36096, 1, 0, 94 ),
                   (0, 64, 0, 0, 0, 384, 1 ),
                   (577536, 0, 9024, 94, 1, 376, 0 )),)
)

tccg_configs.append(tccg_12)

tccg_13 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (84, 84, 84, 96, 96 ),
    strides    = (((677376, 8064, 1, 0, 84 ),
                   (0, 0, 0, 96, 1 ),
                   (7056, 84, 1, 592704, 0 )),)
)

tccg_configs.append(tccg_13)

tccg_14 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (96, 84, 84, 84, 96 ),
    strides    = (((677376, 8064, 1, 0, 84 ),
                   (0, 0, 0, 96, 1 ),
                   (592704, 84, 1, 7056, 0 )),)
)

tccg_configs.append(tccg_14)


tccg_15 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (96, 84, 84, 84, 96 ),
    strides    = (((677376, 8064, 1, 0, 84 ),
                   (0, 0, 0, 96, 1 ),
                   (592704, 7056, 1, 84, 0 )),)
)

tccg_configs.append(tccg_15)


tccg_16 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (96, 96, 84, 84, 84 ),
    strides    = (((677376, 7056, 1, 0, 84 ),
                   (0, 7056, 0, 84, 1 ),
                   (7056, 0, 1, 84, 0 )),)
)

tccg_configs.append(tccg_16)



tccg_17 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.n, etops.dim.k, etops.dim.k, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (4, 6, 6, 6, 64, 94, 64, 64 ),
    strides    = (((13860864, 385024, 0, 2310144, 6016, 1, 0, 94 ),
                   (0, 262144, 9437184, 1572864, 4096, 0, 64, 1 ),
                   (94, 0, 24064, 0, 0, 1, 376, 0 )),)
)

tccg_configs.append(tccg_17)

tccg_18 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.n, etops.dim.k, etops.dim.k, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (4, 4, 6, 6, 64, 94, 64, 94 ),
    strides    = (((13572096, 3393024, 0, 565504, 8836, 1, 0, 94 ),
                   (0, 2310144, 9240576, 385024, 6016, 0, 94, 1 ),
                   (94, 0, 24064, 0, 0, 1, 376, 0 )),)
)

tccg_configs.append(tccg_18)



tccg_19 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.n, etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (6, 6, 4, 64, 4, 94, 94, 64 ),
    strides    = (((9240576, 6016, 0, 144384, 36096, 1, 0, 94 ),
                   (0, 64, 36096, 0, 0, 0, 384, 1 ),
                   (9048064, 0, 35344, 141376, 94, 1, 376, 0 )),)
)

tccg_configs.append(tccg_19)



tccg_20 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.n, etops.dim.m, etops.dim.m, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (6, 4, 6, 64, 4, 94, 94, 94 ),
    strides    = (((9048064, 8836, 0, 141376, 35344, 1, 0, 94 ),
                   (0, 94, 35344, 0, 0, 0, 376, 1 ),
                   (13572096, 0, 35344, 212064, 94, 1, 376, 0 )),)
)

tccg_configs.append(tccg_20)


tccg_21 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (181, 48, 151, 40, 48, 151 ),
    strides    = (((289920, 6040, 0, 1, 0, 40 ),
                   (0, 7248, 347904, 0, 151, 1 ),
                   (40, 0, 347520, 1, 7240, 0 )),)
)

tccg_configs.append(tccg_21)



tccg_22 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (84, 84, 96, 84, 84, 96 ),
    strides    = (((677376, 8064, 0, 1, 0, 84 ),
                   (0, 8064, 677376, 0, 96, 1 ),
                   (84, 0, 592704, 1, 7056, 0 )),)
)

tccg_configs.append(tccg_22)


tccg_23 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (96, 96, 84, 84, 84, 96 ),
    strides    = (((774144, 8064, 0, 1, 0, 84 ),
                   (0, 8064, 774144, 0, 96, 1 ),
                   (592704, 0, 7056, 1, 84, 0 )),)
)

tccg_configs.append(tccg_23)



tccg_24 = etops.TensorOperationConfig(
    backend    = "tpp",
    data_type  = etops.float32,
    prim_first = etops.prim.zero,
    prim_main  = etops.prim.gemm,
    prim_last  = etops.prim.none,
    dim_types  = (etops.dim.m, etops.dim.k, etops.dim.n, etops.dim.m, etops.dim.n, etops.dim.k ),
    exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim ),
    dim_sizes  = (84, 84, 96, 96, 84, 84 ),
    strides    = (((677376, 8064, 0, 1, 0, 96 ),
                   (0, 7056, 592704, 0, 84, 1 ),
                   (96, 0, 677376, 1, 8064, 0 )),)
)

tccg_configs.append(tccg_24)