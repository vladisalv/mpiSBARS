#include "matrix_gomology.h"

MatrixGomology::MatrixGomology(MyMPI me)
    : MatrixMPI<TypeGomology, ulong>(me, "MatrixGomology")
{
}

MatrixGomology::~MatrixGomology()
{
}
