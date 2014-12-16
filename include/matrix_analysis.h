#ifndef __MATRIX_ANALYSIS_HEADER__
#define __MATRIX_ANALYSIS_HEADER__

#include "matrixMPI.h"
#include "types.h"

class MatrixAnalysis : public MatrixMPI<TypeAnalysis, ulong> {
public:
    MatrixAnalysis(MyMPI me);
    ~MatrixAnalysis();
};
#endif /* __MATRIX_ANALYSIS_HEADER__ */
