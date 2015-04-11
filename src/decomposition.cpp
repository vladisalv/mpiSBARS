#include "decomposition.h"

Decomposition::Decomposition(MyMPI me)
    : MatrixMPI<TypeDecomposition, ulong>(me, "Decomposition")
{
}

Decomposition::~Decomposition()
{
}
