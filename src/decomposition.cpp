#include "decomposition.h"

Decomposition::Decomposition(MyMPI me)
    : MatrixMPI<TypeDecomposition>(me)
{
}

Decomposition::~Decomposition()
{
}
