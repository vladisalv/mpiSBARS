#ifndef __COMPARE_GPU_HEADER__
#define __COMPARE_GPU_HEADER__

#include "types.h"
#include "stdio.h"


void compareDecompositionGpu(TypeDecomposition *decompose1, ulong length_decompose1,
                              TypeDecomposition *decompose2, ulong length_decompose2,
                              ulong width, TypeGomology *data, ulong begin,
                              ulong sum_all, double eps);

#endif /* __COMPARE_GPU_HEADER__ */
