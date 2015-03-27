#include "myMPI.h"
#include "options.h"
#include "gpu_computing.h"

#include "sequence.h"
#include "profiling.h"
#include "profile.h"
#include "decompose.h"
#include "decomposition.h"
#include "compare.h"
#include "matrix_gomology.h"
#include "analyze.h"
#include "set_repeats.h"
#include "image.h"

using namespace std;

int main2(int argc, char *argv[])
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
    if (me.isRoot()) {
        DEBUG(opt.debugInfo());
		DEBUG(me.debugInfo());
    }

    GpuComputing gpu(me, opt.gpuMode());

    Sequence sequence1(me), sequence2(me);
    if (opt.downloadSequence()) {
        sequence1.readFile(opt.getFileSequenceLoad1());
        if (!opt.selfMode())
            sequence2.readFile(opt.getFileSequenceLoad2());
        DEBUG(me.rootMessage("load sequence...\n"));
    }
    if (opt.saveSequence()) {
        if (opt.getFileSequenceSave1())
            sequence1.writeFile(opt.getFileSequenceSave1());
        if (opt.getFileSequenceSave2())
            sequence2.writeFile(opt.getFileSequenceSave2());
        DEBUG(me.rootMessage("save sequence...\n"));
    }

    Profile profile1GC(me), profile1GA(me), profile2GC(me), profile2GA(me);
    if (opt.profileMode()) {
        Profiling profiling(me, opt.getLengthWindowProfile());
        if (opt.modeGC())
            profile1GC = profiling.doProfile(sequence1, 'G', 'C');
        if (opt.modeGA())
            profile1GA = profiling.doProfile(sequence1, 'G', 'A');
        sequence1.free();
        if (!opt.selfMode()) {
            if (opt.modeGC())
                profile2GC = profiling.doProfile(sequence2, 'G', 'C');
            if (opt.modeGA())
                profile2GA = profiling.doProfile(sequence2, 'G', 'A');
            sequence2.free();
        }
        DEBUG(me.rootMessage("profiling done...\n"));
        if (opt.saveProfile()) {
			if (opt.getFileProfileSave1GC())
                profile1GC.writeFile(opt.getFileProfileSave1GC());
			if (opt.getFileProfileSave1GA())
                profile1GA.writeFile(opt.getFileProfileSave1GA());
			if (opt.getFileProfileSave2GC())
                profile2GC.writeFile(opt.getFileProfileSave2GC());
			if (opt.getFileProfileSave2GA())
                profile2GA.writeFile(opt.getFileProfileSave2GA());
            DEBUG(me.rootMessage("save profile..\n."));
        }
    }
    if (opt.downloadProfile()) {
        if (opt.getFileProfileLoad1GC())
            profile1GC.readFile(opt.getFileProfileLoad1GC());
        if (opt.getFileProfileLoad1GA())
            profile1GA.readFile(opt.getFileProfileLoad1GA());
        if (opt.getFileProfileLoad2GC())
            profile2GC.readFile(opt.getFileProfileLoad2GC());
        if (opt.getFileProfileLoad2GA())
            profile2GA.readFile(opt.getFileProfileLoad2GA());
        DEBUG(me.rootMessage("load profile...\n"));
    }
    double profile_time = me.getTime() - begin_time;

    Decomposition decomposition1GC(me), decomposition1GA(me);
    Decomposition decomposition2GC(me), decomposition2GA(me);
    if (opt.decomposeMode()) {
        Decompose decompose(me, gpu, opt.getLengthWindowDecompose(),
                                     opt.getStepDecompose(),
                                     opt.getNumberCoefDecompose());
        if (opt.modeGC())
            decomposition1GC = decompose.doDecompose(profile1GC);
        if (opt.modeGA())
            decomposition1GA = decompose.doDecompose(profile1GA);
        profile1GC.free();
        profile1GA.free();
        if (!opt.selfMode()) {
            if (opt.modeGC())
                decomposition2GC = decompose.doDecompose(profile2GC);
            if (opt.modeGA())
                decomposition2GA = decompose.doDecompose(profile2GA);
            profile2GC.free();
            profile2GA.free();
        }
        DEBUG(me.rootMessage("decompose done...\n"));
        if (opt.saveDecompose()) {
            if (opt.getFileDecompositionSave1GC())
                decomposition1GC.writeFile(opt.getFileDecompositionSave1GC());
            if (opt.getFileDecompositionSave1GA())
                decomposition1GA.writeFile(opt.getFileDecompositionSave1GA());
            if (opt.getFileDecompositionSave2GC())
                decomposition2GC.writeFile(opt.getFileDecompositionSave2GC());
            if (opt.getFileDecompositionSave2GA())
                decomposition2GA.writeFile(opt.getFileDecompositionSave2GA());
            DEBUG(me.rootMessage("save decomposition...\n"));
        }
    }
    if (opt.downloadDecompose()) {
        if (opt.getFileDecompositionLoad1GC())
            decomposition1GC.readFile(opt.getFileDecompositionLoad1GC());
        if (opt.getFileDecompositionLoad1GA())
            decomposition1GA.readFile(opt.getFileDecompositionLoad1GA());
        if (opt.getFileDecompositionLoad2GC())
            decomposition2GC.readFile(opt.getFileDecompositionLoad2GC());
        if (opt.getFileDecompositionLoad2GA())
            decomposition2GA.readFile(opt.getFileDecompositionLoad2GA());
        DEBUG(me.rootMessage("load decomposition...\n"));
    }
    double decompose_time = me.getTime() - begin_time;

    MatrixGomology matrixGomologyGC(me), matrixGomologyGA(me), matrixGomology(me);
    Image image(me);
    if (opt.gomologyMode()) {
        Compare compare(me, gpu, opt.getEps());
        if (opt.selfMode()) {
            if (opt.modeGC())
                matrixGomologyGC = compare.doCompare(decomposition1GC);
            if (opt.modeGA())
                matrixGomologyGA = compare.doCompare(decomposition1GA);
        } else {
            if (opt.modeGC())
                matrixGomologyGC = compare.doCompare(decomposition1GC, decomposition2GC);
            if (opt.modeGA())
                matrixGomologyGA = compare.doCompare(decomposition1GA, decomposition2GA);
        }
        decomposition1GC.free();
        decomposition1GA.free();
        decomposition2GC.free();
        decomposition2GA.free();
        if (opt.modeGC() && opt.modeGA()) {
            matrixGomology = compare.comparisonMatrix(matrixGomologyGC, matrixGomologyGA);
            matrixGomologyGA.free();
        }
        else if (opt.modeGC())
            matrixGomology = matrixGomologyGC;
        else if (opt.modeGA())
            matrixGomology = matrixGomologyGA;
        DEBUG(me.rootMessage("compare done...\n"));
        if (opt.saveGomology()) {
            if (opt.getFileMatrixGomologySave())
                matrixGomology.writeFile(opt.getFileMatrixGomologySave());
            if (opt.getFileImageSave())
                image.saveImage(matrixGomology, opt.getFileImageSave());
            DEBUG(me.rootMessage("save matrix gomology...\n"));
        }
    }
    if (opt.downloadGomology()) {
        if (opt.getFileMatrixGomologyLoad())
            matrixGomology.readFile(opt.getFileMatrixGomologyLoad());
        if (opt.getFileImageLoad())
            image.loadImage(matrixGomology, opt.getFileImageLoad());
        DEBUG(me.rootMessage("load matrix gomology...\n"));
    }
    double gomology_time = me.getTime() - begin_time;


    SetRepeats setRepeats(me);
    if (opt.analysisMode()) {
        Analyze analyze(me, gpu, opt.getEps(), opt.getMinLengthRepeat(),
                        opt.getFidelityRepeat(), opt.getLimitMemoryMatrix());
        if (!matrixGomology.isEmpty()) {
            setRepeats = analyze.doAnalyze(matrixGomology);
            matrixGomology.free();
        } else {
            SetRepeats repGC(me), repGA(me);
            if (opt.modeGC()) {
                if (opt.selfMode())
                    repGC = analyze.doAnalyze(decomposition1GC);
                else
                    repGC = analyze.doAnalyze(decomposition1GC, decomposition2GC);
            }
            if (opt.modeGA()) {
                if (opt.selfMode())
                    repGA = analyze.doAnalyze(decomposition1GA);
                else
                    repGA = analyze.doAnalyze(decomposition1GA, decomposition2GA);
            }
            if (opt.modeGC() && opt.modeGA())
                setRepeats = analyze.comparisonRepeats(repGC, repGA);
            decomposition1GC.free();
            decomposition1GA.free();
            decomposition2GC.free();
            decomposition2GA.free();
        }
        setRepeats.analyzeOtherProcess();
        DEBUG(me.rootMessage("analyze repeats done...\n"));
        if (opt.saveAnalysis()) {
            setRepeats.writeFile(opt.getFileAnalysisSave());
            DEBUG(me.rootMessage("save repeats...\n"));
        }
    }
    if (opt.downloadAnalysis()) {
        setRepeats.readFile(opt.getFileAnalysisLoad());
        DEBUG(me.rootMessage("load repeats...\n"));
    }
    double analyze_time = me.getTime() - begin_time;

    double total_time = me.getTime() - begin_time;
    me.rootMessage("Total time = %lf\n", total_time);

    me.rootMessage("profile.   Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", profile_time, profile_time, (profile_time) / total_time * 100);
    me.rootMessage("decompose. Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", decompose_time, decompose_time - profile_time, (decompose_time - profile_time) / total_time * 100);
    me.rootMessage("gomology.  Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", gomology_time, gomology_time - decompose_time, (gomology_time - decompose_time) / total_time * 100);
    me.rootMessage("analyze.   Time = %5.2lf. Clear time = %5.2lf. This is %3.1lf%% of time\n", analyze_time, analyze_time - gomology_time, (analyze_time - gomology_time) / total_time * 100);

    return 0;
}
