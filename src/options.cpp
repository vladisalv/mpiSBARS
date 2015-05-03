#include "options.h"

Options::Options(int argc, char *argv[])
{
    program_name = version_name = version_number = 0;
    help_mode = version_mode = debug_mode = error_mode = false;

    eps = 0.;
    length_window_profile = length_window_decompose = 0;
    step_decompose = number_coef_decompose = 0;
    min_length_repeat = 1;
    fidelity_repeat = 1.;
    limit_memory = 100; // 100 Mb

    profile_mode = decompose_mode = gomology_mode = analysis_mode = false;
    gpu_mode = false;
    self_mode = true;
    gc_mode = ga_mode = true;
    use_matrix = false;

    save_profile = download_profile = false;
    save_profile = download_profile = false;
    save_decompose = download_decompose = false;
    save_gomology = download_gomology = false;
    save_analysis = download_analysis = false;

    sequence_load_first = sequence_load_second = 0;
    sequence_save_first = sequence_save_second = 0;
    profile_load_firstGC = profile_load_secondGC = 0;
    profile_load_firstGA = profile_load_secondGA = 0;
    profile_save_firstGC = profile_save_secondGC = 0;
    profile_save_firstGA = profile_save_secondGA = 0;
    decomposition_load_firstGC = decomposition_load_secondGC = 0;
    decomposition_load_firstGA = decomposition_load_secondGA = 0;
    decomposition_save_firstGC = decomposition_save_secondGC = 0;
    decomposition_save_firstGA = decomposition_save_secondGA = 0;
    matrix_gomology_load = matrix_gomology_save = 0;
    analysis_load = analysis_save = 0;
    image_save = image_load = 0;

    parseOptions(argc, argv);
}


Options::~Options()
{
    return;
}

void Options::parseOptions(int argc, char *argv[])
{
    parse(argc, argv);
    if (help_mode || version_mode)
        return;
    checkOptions();
    setMode();
}

void Options::parse(int argc, char *argv[])
{
#ifdef PROGRAM_NAME
    program_name = PROGRAM_NAME;
#else
    program_name = argv[0];
#endif
#ifdef VERSION
    version_name = VERSION;
#else
    version_name = "UNKNOW";
#endif
#ifdef VERSION_NUMBER
    version_number = VERSION_NUMBER;
#else
    version_number = "UNKNOW";
#endif
    struct option longopts[] = {
        {"help",                    no_argument,       NULL, 'h'},
        {"version",                 no_argument,       NULL, 'v'},
        {"debug-mode",              required_argument, NULL, 'd'},
        {"gpu",                     no_argument,       NULL, 'g'},
        {"gc-only",                 no_argument,       NULL,  25},
        {"ga-only",                 no_argument,       NULL,  26},
        {"use-matrix",              no_argument,       NULL,  27},
        {"limit-memory",            required_argument, NULL,  30},
        {"eps",                     required_argument, NULL, 'e'},
        {"number-coefficient",      required_argument, NULL, 'c'},
        {"profiling-window",        required_argument, NULL, 'w'},
        {"decompose-window",        required_argument, NULL, 'a'},
        {"step-decompose",          required_argument, NULL, 's'},
        {"fidelity-repeat",         required_argument, NULL,  28},
        {"min-length-repeat",       required_argument, NULL,  29},
        {"sequence-load-first",     required_argument, NULL, 'f'},
        {"sequence-load-second",    required_argument, NULL, 'F'},
        {"sequence-save-first",     required_argument, NULL,   2},
        {"sequence-save-second",    required_argument, NULL,   3},
        {"profile-load-firstGC",    required_argument, NULL,   4},
        {"profile-load-firstGA",    required_argument, NULL,   5},
        {"profile-load-secondGC",   required_argument, NULL,   6},
        {"profile-load-secondGA",   required_argument, NULL,   7},
        {"profile-save-firstGC",    required_argument, NULL,   8},
        {"profile-save-firstGA",    required_argument, NULL,   9},
        {"profile-save-secondGC",   required_argument, NULL,  10},
        {"profile-save-secondGA",   required_argument, NULL,  11},
        {"decompose-load-firstGC",  required_argument, NULL,  12},
        {"decompose-load-firstGA",  required_argument, NULL,  13},
        {"decompose-load-secondGC", required_argument, NULL,  14},
        {"decompose-load-secondGA", required_argument, NULL,  15},
        {"decompose-save-firstGC",  required_argument, NULL,  16},
        {"decompose-save-firstGA",  required_argument, NULL,  17},
        {"decompose-save-secondGC", required_argument, NULL,  18},
        {"decompose-save-secondGA", required_argument, NULL,  19},
        {"matrix-gomology-load",    required_argument, NULL,  20},
        {"matrix-gomology-save",    required_argument, NULL,  21},
        {"repeats-analysis-load",   required_argument, NULL,  22},
        {"repeats-analysis-save",   required_argument, NULL, 'o'},
        {"image-save",              required_argument, NULL, 'O'},
        {"image-load",              required_argument, NULL,  24},
        {0, 0, 0, 0}
    };
    int oc;
    int longindex = -1;
    const char *optstring = ":hvgd:e:c:w:a:s:f:F:o:O:"; // opterr = 0, because ":..."
    while ((oc = getopt_long(argc, argv, optstring, longopts, &longindex)) != -1) {
        switch (oc) {
        case 'h':
            help_mode = true;
            break;
        case 'v':
            version_mode = true;
            break;
        case 'g':
            gpu_mode = true;
            break;
        case 25:
            ga_mode = false;
            break;
        case 26:
            gc_mode = false;
            break;
        case 27:
            use_matrix = true;
            break;
        case 30:
            limit_memory = atol(optarg);
            break;
        case 'd':
            debug_mode = true;
            debug_level = atoi(optarg);
            break;
        case 'e':
            eps = atof(optarg);
            break;
        case 'c':
            number_coef_decompose = atoi(optarg);
            break;
        case 'w':
            length_window_profile = atoi(optarg);
            break;
        case 'a':
            length_window_decompose = atoi(optarg);
            break;
        case 's':
            step_decompose = atoi(optarg);
            break;
        case 28:
            fidelity_repeat = atof(optarg);
            break;
        case 29:
            min_length_repeat = atol(optarg);
            break;
        case 'f':
            sequence_load_first = optarg;
            break;
        case 'F':
            sequence_load_second = optarg;
            break;
        case 2:
            sequence_save_first = optarg;
            break;
        case 3:
            sequence_save_second = optarg;
            break;
        case 4:
            profile_load_firstGC = optarg;
            break;
        case 5:
            profile_load_firstGA = optarg;
            break;
        case 6:
            profile_load_secondGC = optarg;
            break;
        case 7:
            profile_load_secondGA = optarg;
            break;
        case 8:
            profile_save_firstGC = optarg;
            break;
        case 9:
            profile_save_firstGA = optarg;
            break;
        case 10:
            profile_save_secondGC = optarg;
            break;
        case 11:
            profile_save_secondGA = optarg;
            break;
        case 12:
            decomposition_load_firstGC = optarg;
            break;
        case 13:
            decomposition_load_firstGA = optarg;
            break;
        case 14:
            decomposition_load_secondGC = optarg;
            break;
        case 15:
            decomposition_load_secondGA = optarg;
            break;
        case 16:
            decomposition_save_firstGC = optarg;
            break;
        case 17:
            decomposition_save_firstGA = optarg;
            break;
        case 18:
            decomposition_save_secondGC = optarg;
            break;
        case 19:
            decomposition_save_secondGA = optarg;
            break;
        case 20:
            matrix_gomology_load = optarg;
            break;
        case 21:
            matrix_gomology_save = optarg;
            break;
        case 22:
            analysis_load = optarg;
            break;
        case 'o':
            analysis_save = optarg;
            break;
        case 'O':
            image_save = optarg;
            break;
        case 24:
            image_load = optarg;
            break;
        case 0: // nothing do
            break;
        case ':':
            error_mode = true; // TODO: error
            break;
        case '?':
        default:
            error_mode = true; // TODO: error
            break;
        }
        longindex = -1;
    }
    //if (optind != argc - 1)
        //error_mode = true; // TODO: error
}

void Options::checkOptions()
{
    checkParameters();
    haveFirst();
    haveDownloadAndSave();
    onlyGCorGA();
}

void Options::checkParameters()
{
    if (!eps || !length_window_profile || !length_window_decompose ||
        !step_decompose || !number_coef_decompose
    ) {
        error_mode = true; // TODO: error
        if (!eps) {
            ;
        }
        if (!length_window_profile) {
            ;
        }
        if (!length_window_decompose) {
            ;
        }
        if (!step_decompose) {
            ;
        }
        if (!number_coef_decompose) {
            ;
        }
    }
}

void Options::haveFirst()
{
    // if have second, but no first
    if (sequence_load_second && !sequence_load_first) {
        error_mode = true;
    }
    if (sequence_save_second && !sequence_save_first) {
        error_mode = true;
    }
    if (profile_load_secondGC && !profile_load_firstGC) {
        error_mode = true;
    }
    if (profile_load_secondGA && !profile_load_firstGA) {
        error_mode = true;
    }
    if (profile_save_secondGC && !profile_save_firstGC) {
        error_mode = true;
    }
    if (profile_save_secondGA && !profile_save_firstGA) {
        error_mode = true;
    }
    if (decomposition_load_secondGC && !decomposition_load_firstGC) {
        error_mode = true;
    }
    if (decomposition_load_secondGA && !decomposition_load_firstGA) {
        error_mode = true;
    }
    if (decomposition_save_secondGC && !decomposition_save_firstGC) {
        error_mode = true;
    }
    if (decomposition_save_secondGA && !decomposition_save_firstGA) {
        error_mode = true;
    }
}

void Options::haveDownloadAndSave()
{
    if (!sequence_load_first && !profile_load_firstGC && !profile_load_firstGA &&
        !decomposition_load_firstGC && !decomposition_load_firstGA &&
        !matrix_gomology_load && !image_load && !analysis_load) {
        error_mode = true;
    }
    if (!sequence_save_first && !profile_save_firstGC && !profile_save_firstGA &&
        !decomposition_save_firstGC && !decomposition_save_firstGA &&
        !matrix_gomology_save && !image_save && !analysis_save) {
        error_mode = true;
    }
}

void Options::onlyGCorGA()
{
    if (!gc_mode && (profile_load_firstGC || decomposition_load_firstGC ||
                     profile_save_firstGC || decomposition_save_firstGC)) {
        error_mode = true;
    }
    if (!ga_mode && (profile_load_firstGA || decomposition_load_firstGA ||
                     profile_save_firstGA || decomposition_save_firstGA)) {
        error_mode = true;
    }
}


void Options::setMode()
{
    defineDownload();
    defineSave();
    defineMode();
}

void Options::defineDownload()
{
    // if two load data from other level
    bool input_data = false;
    if (sequence_load_first) {
        input_data = true;
        download_sequence = true;
        if (sequence_load_second)
            self_mode = false;
    }
    if ((profile_load_firstGC || profile_load_firstGA) && input_data) {
        error_mode = true;
    } else if (profile_load_firstGC || profile_load_firstGA) {
        input_data = true;
        download_profile = true;
        if (profile_load_secondGC || profile_load_secondGA)
            self_mode = false;
    }
    if ((decomposition_load_firstGC || decomposition_load_firstGA) && input_data) {
        error_mode = true;
    } else if (decomposition_load_firstGC || decomposition_load_firstGA) {
        input_data = true;
        download_decompose = true;
        if (decomposition_load_secondGC || decomposition_load_secondGA)
            self_mode = false;
    }
    if ((matrix_gomology_load || image_load) && input_data) {
        error_mode = true;
    } else if (matrix_gomology_load || image_load) {
        input_data = true;
        download_gomology = true;
    }
    if (analysis_load && input_data) {
        error_mode = true;
    } else if (analysis_load) {
        input_data = true;
        download_analysis = true;
    }
}

void Options::defineSave()
{
    bool output_data = false;
    if (sequence_save_first) {
        save_sequence = true;
        output_data = true;
        if (!download_sequence) {
            error_mode = true;
        }
        if (!self_mode && !sequence_load_first) {
            error_mode = true;
        }
    }
    if (profile_save_firstGC || profile_save_firstGA) {
        save_profile = true;
        output_data = true;
        if (!download_sequence) {
            error_mode = true;
        }
        if (!self_mode && (!profile_save_secondGC || !profile_save_secondGA)) {
            error_mode = true;
        }
    }
    if (decomposition_save_firstGC || decomposition_save_firstGA) {
        save_decompose = true;
        output_data = true;
        if (!download_sequence && !download_profile) {
            error_mode = true;
        }
        if (!self_mode && (!decomposition_save_secondGC || !decomposition_save_secondGA)) {
            error_mode = true;
        }
    }
    if (matrix_gomology_save || image_save) {
        save_gomology = true;
        output_data = true;
        if (!download_sequence && !download_profile && !download_decompose) {
            error_mode = true;
        }
    }
    if (analysis_save) {
        save_analysis = true;
        output_data = true;
        if (!download_sequence && !download_profile && !download_decompose &&
            !download_gomology
        ) {
            error_mode = true;
        }
    }
    if (!output_data) { // it's unreal
        error_mode = true;
    }
}

void Options::defineMode()
{
    if (download_sequence && (save_profile || save_decompose || save_gomology ||
                                save_analysis))
        profile_mode = true;
    if ((download_sequence || download_profile) && (save_decompose || save_gomology ||
                                                    save_analysis))
        decompose_mode = true;
    if ((download_sequence || download_profile || download_decompose) &&
        (save_gomology || use_matrix))
        gomology_mode = true;
    if ((download_sequence || download_profile || download_decompose || download_gomology) &&
        (save_analysis))
        analysis_mode = true;
}



void Options::debugInfo(const char *file, int line, const char *info)
{
    /*
    printf("\n");
    printf("Information about options:\n");
    printf("length_window_profile = %d\n", length_window_profile);
    printf("length_window_decompose = %d\n", length_window_decompose);
    printf("step_decompose = %d\n", step_decompose);
    printf("number_coef_decompose = %d\n", number_coef_decompose);
    printf("eps = %f\n", eps);
    */
    printf("This is debugInfo(%s) of %s in %s at line %d\n", info, "Options", file, line);
    printf("name f = %s\n", sequence_load_first);
    printf("name F = %s\n", sequence_load_second);
    printf("use gpu = %s\n", gpu_mode ? "true" : "false");
    printf("\n");
    if (download_sequence)
        printf("download_sequence\n");
    if (save_sequence)
        printf("\t\t save_sequence\n");
    if (profile_mode)
        printf("\t profile_mode\n");
    if (save_profile)
        printf("\t\t save_profile\n");
    if (download_profile)
        printf("download_profile\n");
    if (decompose_mode)
        printf("\t decompose_mode\n");
    if (save_decompose)
        printf("\t\t save_decompose\n");
    if (download_decompose)
        printf("download_decompose\n");
    if (gomology_mode)
        printf("\t gomology_mode\n");
    if (save_gomology)
        printf("\t\t save_gomology\n");
    if (download_gomology)
        printf("download_gomology\n");
    if (analysis_mode)
        printf("\t analysis_mode\n");
    if (save_analysis)
        printf("\t\t save_analysis\n");
    if (download_analysis)
        printf("download_analysis\n");
}


void Options::helpPrint()
{
    printf("Help\n");
    /*
    GC_LOG((
        " GC_COUNT v0.2\n"
        " USAGE: ./gc_count [-h] [-l] [-d] [-w N] [-a N] [-s N] [-c N] [-e E]\n"
        "                   [-m N] -QWER <abhkxz[d|i|t][:N]> [-p F] [-P F] -i <DNA_FILE>\n"
        "                   [-I <DNA_FILE2>] -o <IMAGE_FILE>\n"
            "Input and output options:\n"
        " -i F : use file with name F as first DNA sequence file\n"
        " -I F : use file with name F as second DNA sequence file\n"
        " -l   : only profile\n"
        " -p F : write profile of first sequence into file with name F\n"
        " -P F : write profile of second sequence into file with name F\n"
        " -o F : write output picture to file with name F (%s, \'-\' means stdout)\n\n"
        "Approximation options:\n"
        " -w N : set profile window size to N (default %d)\n"
        " -a N : set approximation window size to N (default %d)\n"
        " -s N : set approximation step to N (default %d)\n"
        " -c N : set decomposition depth to N (default %d)\n"
        " -e E : set epsilon to E (default %.3g)\n\n"
            "Find repeats options:\n"
        " -QWER abhkxz[d|i|t]:N : mask of repeats\n"
        "                       : a - gc direct\n"
        "                       : b - gc invert\n"
        "                       : h - ga direct\n"
        "                       : k - ga invert\n"
        "                       : x - complementary direct\n"
        "                       : z - complementary invert\n"
        "                       : d - direction key for search in main diagonal\n"
        "                       : i - direction key for search in sub diagonal\n"
        "                       : t - direction key for search tandem repeats\n"
        " -d   : print debug messages\n"
        " -h   : print this help message and exit\n\n",
        AVAILABLE_FORMATS,DEFAULT_WINDOW_GC, DEFAULT_WINDOW_AP, DEFAULT_STEP,
        DEFAULT_COEF, DEFAULT_EPS));
    */
}

void Options::versionPrint()
{
    printf("Version\n");
}

void Options::errorPrint()
{
    printf("Error\n");
}
// ==================  FUNCTION MODE  ===============================================

const char* Options::getProgramName()
{
    return program_name;
}

const char* Options::getProgramVersion()
{
    return version_name;
}

const char* Options::getProgramVersionNumber()
{
    return version_number;
}


unsigned int Options::getDebugLevel()
{
    return debug_level;
}



bool Options::helpMode()
{
    return help_mode;
}

bool Options::versionMode()
{
    return version_mode;
}

bool Options::errorMode()
{
    return error_mode;
}

bool Options::debugMode()
{
    return debug_mode;
}



double Options::getEps()
{
    return eps;
}

unsigned int Options::getLengthWindowProfile()
{
    return length_window_profile;
}

unsigned int Options::getLengthWindowDecompose()
{
    return length_window_decompose;
}

unsigned int Options::getStepDecompose()
{
    return step_decompose;
}

unsigned int Options::getNumberCoefDecompose()
{
    return number_coef_decompose;
}

double Options::getFidelityRepeat()
{
    return fidelity_repeat;
}

unsigned long Options::getMinLengthRepeat()
{
    return min_length_repeat;
}

size_t Options::getLimitMemoryMatrix()
{
    return limit_memory; // limit_memory MB
    //return 1048576 * limit_memory; // limit_memory MB
}



bool Options::profileMode()
{
    return profile_mode;
}

bool Options::decomposeMode()
{
    return decompose_mode;
}

bool Options::gomologyMode()
{
    return gomology_mode;
}

bool Options::analysisMode()
{
    return analysis_mode;
}




bool Options::gpuMode()
{
    return gpu_mode;
}

bool Options::selfMode()
{
    return self_mode;
}

bool Options::modeGC()
{
    return gc_mode;
}

bool Options::modeGA()
{
    return ga_mode;
}




bool Options::saveSequence()
{
    return save_sequence;
}

bool Options::downloadSequence()
{
    return download_sequence;
}

bool Options::saveProfile()
{
    return save_profile;
}

bool Options::downloadProfile()
{
    return download_profile;
}

bool Options::saveDecompose()
{
    return save_decompose;
}

bool Options::downloadDecompose()
{
    return download_decompose;
}

bool Options::saveGomology()
{
    return save_gomology;
}

bool Options::downloadGomology()
{
    return download_gomology;
}

bool Options::saveAnalysis()
{
    return save_analysis;
}

bool Options::downloadAnalysis()
{
    return download_analysis;
}





char* Options::getFileSequenceLoad1()
{
    return sequence_load_first;
}
char* Options::getFileSequenceLoad2()
{
    return sequence_load_second;
}
char* Options::getFileSequenceSave1()
{
    return sequence_save_first;
}
char* Options::getFileSequenceSave2()
{
    return sequence_save_second;
}
char* Options::getFileProfileLoad1GC()
{
    return profile_load_firstGC;
}
char* Options::getFileProfileLoad1GA()
{
    return profile_load_firstGA;
}
char* Options::getFileProfileLoad2GC()
{
    return profile_load_secondGC;
}
char* Options::getFileProfileLoad2GA()
{
    return profile_load_secondGA;
}
char* Options::getFileProfileSave1GC()
{
    return profile_save_firstGC;
}
char* Options::getFileProfileSave1GA()
{
    return profile_save_firstGA;
}
char* Options::getFileProfileSave2GC()
{
    return profile_save_secondGC;
}
char* Options::getFileProfileSave2GA()
{
    return profile_save_secondGA;
}
char* Options::getFileDecompositionLoad1GC()
{
    return decomposition_load_firstGC;
}
char* Options::getFileDecompositionLoad1GA()
{
    return decomposition_load_firstGA;
}
char* Options::getFileDecompositionLoad2GC()
{
    return decomposition_load_secondGC;
}
char* Options::getFileDecompositionLoad2GA()
{
    return decomposition_load_secondGA;
}
char* Options::getFileDecompositionSave1GC()
{
    return decomposition_save_firstGC;
}
char* Options::getFileDecompositionSave1GA()
{
    return decomposition_save_firstGA;
}
char* Options::getFileDecompositionSave2GC()
{
    return decomposition_save_secondGC;
}
char* Options::getFileDecompositionSave2GA()
{
    return decomposition_save_secondGA;
}
char* Options::getFileMatrixGomologyLoad()
{
    return matrix_gomology_load;
}
char* Options::getFileMatrixGomologySave()
{
    return matrix_gomology_save;
}
char* Options::getFileAnalysisLoad()
{
    return analysis_load;
}
char* Options::getFileAnalysisSave()
{
    return analysis_save;
}
char* Options::getFileImageLoad()
{
    return image_save;
}
char* Options::getFileImageSave()
{
    return image_save;
}
// ==================  END MODE  ===============================================
