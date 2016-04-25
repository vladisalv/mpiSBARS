#include "matrix_gomology.h"

MatrixGomology::MatrixGomology(MyMPI me)
    : MatrixMPI<TypeGomology>(me, "MatrixGomology", MPI_TYPE_GOMOLOGY)
{
}

MatrixGomology::~MatrixGomology()
{
}
