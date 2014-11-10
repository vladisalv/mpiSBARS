#include "analyze.h"

Analyze::Analyze(MyMPI new_me)
    : me(new_me)
{
}

Analyze::~Analyze()
{
}


MatrixAnalysis Analyze::doAnalyze(MatrixGomology matrixGomology)
{
    MatrixAnalysis matrixAnalysis(me);
    matrixAnalysis.data = matrixGomology.data;
    matrixAnalysis.length = matrixGomology.length;
    matrixAnalysis.width = matrixGomology.width;
    matrixAnalysis.height = matrixGomology.height;
    return matrixAnalysis;
}
