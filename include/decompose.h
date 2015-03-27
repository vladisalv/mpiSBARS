#ifndef __DECOMPOSE_HEADER__
#define __DECOMPOSE_HEADER__

#include "gpu_computing.h"
#include "profile.h"
#include "decomposition.h"
#include "types.h"

#include <math.h> // ceil()

class Decompose {
    MyMPI me;
    GpuComputing gpu;

    uint window, step, number_coef;

    void decomposeFourier(TypeDecomposition *u, uint m, TypeProfile *y, uint k);
public:
    Decompose(MyMPI me, GpuComputing gpu, uint window, uint step, uint number_coef);
    ~Decompose();

    Decomposition doDecompose(Profile &profile);

    uint getLengthWindowDecompose();
    uint getStepDecompose();
    uint getNumberCoefDecompose();

    void setLengthWindowDecompose(uint window_new);
    void setStepDecompose(uint step_new);
    void setNumberCoefDecompose(uint number_coef_new);
};
#endif /* __DECOMPOSE_HEADER__ */
