PROGRAM_NAME := mpiSSSRDNA
VERSION = $(TARGET_NOW)
VERSION_NUMBER := 3.0
TARGET_NOW := debug
#TARGET_NOW := release

PROFILING_WINDOW   = 250
DECOMPOSE_WINDOW   = 250
STEP_DECOMPOSE     = 100
NUMBER_COEFFICIENT = 75
EPS                = 0.01

INPUT_DIR := ./test/input
#INPUT_FILE1 := rat165M
#INPUT_FILE1 := file20M
INPUT_FILE1 := file5M
#INPUT_FILE1 := file50K
#INPUT_FILE1 := $(FILE)

OUTPUT_DIR     := ./test/output
OUTPUT_FILE     = out_$(N)_$(DATE)_$(GPU)
OUTPUT_FILE_DEC = dec_$(DATE)_$(GPU)
OUTPUT_FILE_GOM = gom_$(DATE)_$(GPU)
OUTPUT_FILE_PIC = pic_$(DATE)_$(GPU)
OUTPUT_FILE_ANA = ana_$(DATE)_$(GPU)

GPU = --gpu
#USE_MATRIX = --use-matrix
SEQUENCE = -f $(INPUT_DIR)/$(INPUT_FILE1)                              \
           #-F $(INPUT_DIR)/$(INPUT_FILE2)
#DEC = --decompose-save-firstGC $(OUTPUT_DIR)/$(OUTPUT_FILE_DEC)_GC \
      --decompose-save-firstGA $(OUTPUT_DIR)/$(OUTPUT_FILE_DEC)_GA
#GOMOLOGY = --matrix-gomology-save $(OUTPUT_DIR)/$(OUTPUT_FILE_GOM)
#PIC = --image-save $(OUTPUT_DIR)/$(OUTPUT_FILE_PIC)
ANALYS = --repeats-analysis-save $(OUTPUT_DIR)/$(OUTPUT_FILE_ANA)

#GC = --gc-only
#GA = --ga-only
BLOCK = 10000

ARGUMENTS = --profiling-window     $(PROFILING_WINDOW)                  \
            --decompose-window     $(DECOMPOSE_WINDOW)                  \
            --step-decompose       $(STEP_DECOMPOSE)                    \
            --number-coefficient   $(NUMBER_COEFFICIENT)                \
            --eps                  $(EPS)                               \
            $(GPU) $(USE_MATRIX) $(SEQUENCE) $(DEC) $(GOMOLOGY) $(PIC) $(ANALYS) $(GC) $(GA) --limit-memory $(BLOCK)

# HOST, MPI, LOMONOSOV or BLUEGENE
MACHINE := MPI

# short name NUMBER_PROC
N           = 4
NUMBER_PROC = $(N)

#short name NUMBER_NODE
NN          = 1
NT          = 8
NUMBER_NODE = $(NN)
NODE_TASK   = $(NT)
QUEUE       = gputest
TIME        = 15:00

#define mode compile
USE_CUDA = 1
USE_MPI  = 1

#redefine compiler
ifdef USE_MPI
    CXX = mpicxx
endif
