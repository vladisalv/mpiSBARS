#include "decomposition.h"

Decomposition::Decomposition(MyMPI me)
    : MatrixMPI<TypeDecomposition>(me, "Decomposition", MPI_TYPE_DECOMPOSITION)
{
}

Decomposition::~Decomposition()
{
}
