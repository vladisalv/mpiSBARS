#ifndef __OPTIONS_HEADER__
#define __OPTIONS_HEADER__

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

class Options {
    bool profile_mode, decompose_mode, gomology_mode, analysis_mode, draw_mode;
    bool help_mode, version_mode, debug_mode, error_mode;
    bool gpu_mode, self_mode;
    bool save_sequence, download_sequence;
    bool save_profile, download_profile;
    bool save_decompose, download_decompose;
    bool save_gomology, download_gomology;
    bool save_analysis, download_analysis;

    const char *program_name;
    const char *version_name, *version_number;

    unsigned int debug_level;

    char *sequence_load_first, *sequence_load_second;
    char *sequence_save_first, *sequence_save_second;
    char *profile_load_firstGC, *profile_load_secondGC;
    char *profile_load_firstGA, *profile_load_secondGA;
    char *profile_save_firstGC, *profile_save_secondGC;
    char *profile_save_firstGA, *profile_save_secondGA;
    char *decomposition_load_firstGC, *decomposition_load_secondGC;
    char *decomposition_load_firstGA, *decomposition_load_secondGA;
    char *decomposition_save_firstGC, *decomposition_save_secondGC;
    char *decomposition_save_firstGA, *decomposition_save_secondGA;
    char *matrix_gomology_load, *matrix_gomology_save;
    char *matrix_analysis_load, *matrix_analysis_save;
    char *output_file;

    double eps; // fidelity of compute
    unsigned int length_window_profile;    // length window of profiling
    unsigned int length_window_decompose;  // length window of decomposition
    unsigned int step_decompose; // step window of approximation
    unsigned int number_coef_decompose; // number coefficient of decomposition


    void parseOptions(int argc, char *argv[]);
    void parse(int argc, char *argv[]);

    void checkOptions();
    void checkParameters();
    void haveFirst();
    void haveGCandGA();

    void setMode();
    void defineDownload();
    void defineSave();
    void defineMode();
public:
    Options(int argc, char *argv[]);
    ~Options();

    void info();

    void helpPrint();
    void versionPrint();
    void errorPrint();

    const char* getProgramName();
    const char* getProgramVersion();
    const char* getProgramVersionNumber();

    unsigned int getDebugLevel();

    bool helpMode();
    bool versionMode();
    bool errorMode();
    bool debugMode();

    double getEps();
    unsigned int getLengthWindowProfile();
    unsigned int getLengthWindowDecompose();
    unsigned int getStepDecompose();
    unsigned int getNumberCoefDecompose();

    bool profileMode();
    bool decomposeMode();
    bool gomologyMode();
    bool analysisMode();
    bool drawMode();

    bool gpuMode();
    bool selfMode();

    bool downloadSequence();
    bool saveSequence();
    bool saveProfile();
    bool downloadProfile();
    bool saveDecompose();
    bool downloadDecompose();
    bool saveGomology();
    bool downloadGomology();
    bool saveAnalysis();
    bool downloadAnalysis();

    char *getFileSequenceLoad1();
    char *getFileSequenceLoad2();
    char *getFileSequenceSave1();
    char *getFileSequenceSave2();
    char *getFileProfileLoad1GC();
    char *getFileProfileLoad1GA();
    char *getFileProfileLoad2GC();
    char *getFileProfileLoad2GA();
    char *getFileProfileSave1GC();
    char *getFileProfileSave1GA();
    char *getFileProfileSave2GC();
    char *getFileProfileSave2GA();
    char *getFileDecompositionLoad1GC();
    char *getFileDecompositionLoad1GA();
    char *getFileDecompositionLoad2GC();
    char *getFileDecompositionLoad2GA();
    char *getFileDecompositionSave1GC();
    char *getFileDecompositionSave1GA();
    char *getFileDecompositionSave2GC();
    char *getFileDecompositionSave2GA();
    char *getFileMatrixGomologyLoad();
    char *getFileMatrixGomologySave();
    char *getFileMatrixAnalysisLoad();
    char *getFileMatrixAnalysisSave();
    char *getFileOutput();
};

#endif /* __OPTIONS_HEADER__*/
