#include "myMPI.h"
#include "options.h"

#include "sequence.h"
#include "profiling.h"
#include "profile.h"
#include "decompose.h"
#include "decomposition.h"
#include "compare.h"
#include "matrix_gomology.h"
#include "analyze.h"
#include "matrix_analysis.h"
#include "image.h"

using namespace std;

int main(int argc, char *argv[])
{
    MyMPI me(MPI_COMM_WORLD, argc, argv);
    double begin_time = me.getTime();

    Options opt(argc, argv);
    if (opt.errorMode()) {
        if (me.isRoot()) {
            opt.errorPrint();
            opt.helpPrint();
        }
        return 1;
    } else if (opt.versionMode()) {
        if (me.isRoot())
            opt.versionPrint();
        return 0;
    } else if (opt.helpMode()) {
        if (me.isRoot())
            opt.helpPrint();
        return 0;
    }
    if (me.isRoot())
        DEBUG(opt.info();)

    Sequence sequence1(me), sequence2(me);
    if (opt.downloadSequence()) {
        sequence1.readFile(opt.getFileSequenceLoad1());
        if (!opt.selfMode())
            sequence2.readFile(opt.getFileSequenceLoad2());
    }

    Profile profile1GC(me), profile1GA(me), profile2GC(me), profile2GA(me);
    if (opt.profileMode()) {
        Profiling profiling(me);
        profile1GC = profiling.doProfile(sequence1, 'G', 'C', opt.getLengthWindowProfile());
        profile1GA = profiling.doProfile(sequence1, 'G', 'A', opt.getLengthWindowProfile());
        sequence1.free();
        if (!opt.selfMode()) {
            profile2GC = profiling.doProfile(sequence2, 'G', 'C', opt.getLengthWindowProfile());
            profile2GA = profiling.doProfile(sequence2, 'G', 'A', opt.getLengthWindowProfile());
            sequence2.free();
        }
        if (opt.saveProfile()) {
            profile1GC.writeFile(opt.getFileProfileSave1GC());
            profile1GA.writeFile(opt.getFileProfileSave1GA());
            if (!opt.selfMode()) {
                profile2GC.writeFile(opt.getFileProfileSave2GC());
                profile2GA.writeFile(opt.getFileProfileSave2GA());
            }
        }
    }
    if (opt.downloadProfile()) {
        profile1GC.readFile(opt.getFileProfileLoad1GC());
        profile1GA.readFile(opt.getFileProfileLoad1GA());
        if (!opt.selfMode()) {
            profile2GC.readFile(opt.getFileProfileLoad2GC());
            profile2GA.readFile(opt.getFileProfileLoad2GA());
        }
    }
    double profile_time = me.getTime() - begin_time;

    Decomposition decomposition1GC(me), decomposition1GA(me);
    Decomposition decomposition2GC(me), decomposition2GA(me);
    if (opt.decomposeMode()) {
        Decompose decompose(me);
        decomposition1GC = decompose.doDecompose(profile1GC, opt.getLengthWindowDecompose(),
            opt.getStepDecompose(), opt.getNumberCoefDecompose(), opt.gpuMode());
        decomposition1GA = decompose.doDecompose(profile1GA, opt.getLengthWindowDecompose(),
            opt.getStepDecompose(), opt.getNumberCoefDecompose(), opt.gpuMode());
        profile1GC.free();
        profile1GA.free();
        if (!opt.selfMode()) {
            decomposition2GC = decompose.doDecompose(profile2GC, opt.getLengthWindowDecompose(),
                opt.getStepDecompose(), opt.getNumberCoefDecompose(), opt.gpuMode());
            decomposition2GA = decompose.doDecompose(profile2GA, opt.getLengthWindowDecompose(),
                opt.getStepDecompose(), opt.getNumberCoefDecompose(), opt.gpuMode());
            profile2GC.free();
            profile2GA.free();
        }
        if (opt.saveDecompose()) {
            decomposition1GC.writeFile(opt.getFileDecompositionSave1GC());
            decomposition1GA.writeFile(opt.getFileDecompositionSave1GA());
            if (!opt.selfMode()) {
                decomposition2GC.writeFile(opt.getFileDecompositionSave2GC());
                decomposition2GA.writeFile(opt.getFileDecompositionSave2GA());
            }
        }
    }
    if (opt.downloadDecompose()) {
        decomposition1GC.readFile(opt.getFileDecompositionLoad1GC());
        decomposition1GA.readFile(opt.getFileDecompositionLoad1GA());
        if (!opt.selfMode()) {
            decomposition2GC.readFile(opt.getFileDecompositionLoad2GC());
            decomposition2GA.readFile(opt.getFileDecompositionLoad2GA());
        }
    }
    double decompose_time = me.getTime() - begin_time;

    MatrixGomology matrixGomology(me);
    if (opt.gomologyMode()) {
        Compare compare(me);
        if (opt.selfMode()) {
            matrixGomology = compare.doCompare(decomposition1GC, decomposition1GA,
                                                opt.getEps());
            decomposition1GC.free();
            decomposition1GA.free();
        } else {
            matrixGomology = compare.doCompare(decomposition1GC, decomposition1GA,
                                               decomposition2GC, decomposition2GA,
                                                opt.getEps());
            decomposition2GC.free();
            decomposition2GA.free();
        }
        if (opt.saveGomology())
            matrixGomology.writeFile(opt.getFileMatrixGomologySave());
    }
    if (opt.downloadGomology())
        matrixGomology.readFile(opt.getFileMatrixGomologyLoad());
    double compare_time = me.getTime() - begin_time;

    MatrixAnalysis matrixAnalysis(me);
    if (opt.analysisMode()) {
        Analyze analyze(me);
        matrixAnalysis = analyze.doAnalyze(matrixGomology);
        //matrixGomology.free();
        if (opt.saveAnalysis())
            matrixAnalysis.writeFile(opt.getFileMatrixAnalysisSave());
    }
    if (opt.downloadAnalysis())
        matrixAnalysis.readFile(opt.getFileMatrixAnalysisLoad());
    double analyze_time = me.getTime() - begin_time;

    Image image(me);
    if (opt.drawMode())
        image.drawImage(matrixAnalysis, opt.getFileOutput());
    double draw_time = me.getTime() - begin_time;

    double total_time = me.getTime() - begin_time;
    me.rootMessage("Total time = %lf\n", total_time);

    me.rootMessage("profile.   Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", profile_time, profile_time, profile_time / total_time * 100);
    me.rootMessage("decompose. Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", decompose_time, decompose_time - profile_time, (decompose_time - profile_time) / total_time * 100);
    me.rootMessage("compare.   Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", compare_time, compare_time - decompose_time, (compare_time - decompose_time) / total_time * 100);
    me.rootMessage("analyze.   Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", analyze_time, analyze_time - compare_time, (analyze_time - compare_time) / total_time * 100);
    me.rootMessage("draw.      Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", draw_time, (draw_time - analyze_time), (draw_time - analyze_time) / total_time * 100);

    return 0;
}
