#ifndef __COMPARE_HEADER__
#define __COMPARE_HEADER__

#include "decomposition.h"
#include "matrix_gomology.h"

class Compare {
    MyMPI me;
    double eps;

    MatrixGomology compareSelf(Decomposition &decomposition);
    MatrixGomology compareTwo(Decomposition &decomposition1, Decomposition &decomposition2);

    MatrixGomology comparisonMatrix(MatrixGomology mat1, MatrixGomology mat2);

    void compareDecomposition(TypeDecomposition *decompose1, ulong length_decompose1,
                              TypeDecomposition *decompose2, ulong length_decompose2,
                              ulong width, TypeGomology *data, ulong begin, ulong sum_all);
    bool compareVector(TypeDecomposition *vec1, TypeDecomposition *vec2, ulong length);
public:
    Compare(MyMPI me);
    ~Compare();

    MatrixGomology doCompare(Decomposition &decompose1GC, Decomposition &decompose1GA, double eps);
    MatrixGomology doCompare(Decomposition &decompose1GC, Decomposition &decompose1GA,
                              Decomposition &decompose2GC, Decomposition &decompose2GA, double eps);
};

#endif /* __COMPARE_HEADER__ */
