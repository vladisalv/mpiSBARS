#include "matrix_analysis.h"

MatrixAnalysis::MatrixAnalysis(MyMPI me)
    : MatrixMPI<TypeAnalysis, ulong>(me)
{
}

MatrixAnalysis::~MatrixAnalysis()
{
}
