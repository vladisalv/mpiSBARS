#ifndef __DECOMPOSITION_HEADER__
#define __DECOMPOSITION_HEADER__

#include "types.h"
#include "matrixMPI.h"

class Decomposition : public MatrixMPI<TypeDecomposition> {
public:
    Decomposition(MyMPI me);
    ~Decomposition();
    friend class Decompose;
    friend class Compare;
    friend class Analyze;
};

#endif /* __DECOMPOSITION_HEADER__ */
