#ifndef __DECOMPOSE_HEADER__
#define __DECOMPOSE_HEADER__

#include "myMPI.h"
#include "profile.h"
#include "decomposition.h"
#include "decomposition_gpu.h"
#include "types.h"

#include <math.h> // ceil()

class Decompose {
    MyMPI me;

    void decomposeFourier(TypeDecomposition *u, uint m, TypeProfile *y, uint k);
public:
    Decompose(MyMPI me);
    ~Decompose();

    Decomposition doDecompose(Profile &profile, uint window, uint step, uint number_coef, bool use_gpu);
};
#endif /* __DECOMPOSE_HEADER__ */
