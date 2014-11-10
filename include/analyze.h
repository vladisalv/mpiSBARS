#ifndef __ANALYZE_HEADER__
#define __ANALYZE_HEADER__

#include "myMPI.h"
#include "matrix_gomology.h"
#include "matrix_analysis.h"

class Analyze {
    MyMPI me;
public:
    Analyze(MyMPI me);
    ~Analyze();

    MatrixAnalysis doAnalyze(MatrixGomology matrixGomology);
};

#endif /* __ANALYZE_HEADER__ */
