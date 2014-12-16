#ifndef __DECOMPOSITION_GPU_HEADER__
#define __DECOMPOSITION_GPU_HEADER__

#include "types.h"

#include <stdio.h>
#include <stdlib.h>

void doDecomposeGPU(TypeDecomposition *dataHost, uint number_window, uint number_coef,
                    TypeProfile *profileHost, uint window, uint step);

#endif /* __DECOMPOSITION_GPU_HEADER__ */
