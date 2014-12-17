#ifndef __DECOMPOSE_HEADER__
#define __DECOMPOSE_HEADER__

#include "gpu_computing.h"
#include "profile.h"
#include "decomposition.h"
#include "types.h"

#include <math.h> // ceil()

class Decompose {
    MyMPI me;

    void decomposeFourier(TypeDecomposition *u, uint m, TypeProfile *y, uint k);
public:
    Decompose(MyMPI me);
    ~Decompose();

    Decomposition doDecompose(Profile &profile, uint window, uint step, uint number_coef, GpuComputing gpu);
};
#endif /* __DECOMPOSE_HEADER__ */
