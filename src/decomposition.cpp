#include "decomposition.h"

Decomposition::Decomposition(MyMPI me)
    : MatrixMPI<double, long unsigned int>(me)
{
}

Decomposition::~Decomposition()
{
}
