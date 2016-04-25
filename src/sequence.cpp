#include "sequence.h"

Sequence::Sequence(MyMPI me)
    : ArrayMPI<TypeSequence>(me, "Sequence", MPI_TYPE_SEQUENCE)
{
}

Sequence::~Sequence()
{
}
