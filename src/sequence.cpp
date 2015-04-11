#include "sequence.h"

Sequence::Sequence(MyMPI me)
    : ArrayMPI<TypeSequence, ulong>(me, "Sequence")
{
}

Sequence::~Sequence()
{
}
