#ifndef __MATRIX_GOMOLOGYY_HEADER__
#define __MATRIX_GOMOLOGYY_HEADER__

#include "types.h"
#include "matrixMPI.h"

class MatrixGomology : public MatrixMPI<TypeGomology, ulong> {
public:
    MatrixGomology(MyMPI me);
    ~MatrixGomology();
    friend class Compare;
    friend class Image;
    friend class Analyze;
};

#endif /* __MATRIX_GOMOLOGYY_HEADER__ */
