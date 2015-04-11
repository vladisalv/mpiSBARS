PROGRAM_NAME  := SSSDNA
VERSION_NUMER := 3.0
DEBUG_MODE    := 1

PROFILING_WINDOW   = 250
DECOMPOSE_WINDOW   = 250
STEP_DECOMPOSE     = 100
NUMBER_COEFFICIENT = 75
EPS                = 0.01
INPUT_DIR := ./test/samples
#INPUT_FILE := rat165M
#INPUT_FILE1 := file7.5M
#INPUT_FILE1 := file5M
#INPUT_FILE2 := file5M
INPUT_FILE1 := file50K
#INPUT_FILE2 := file50K

OUTPUT_DIR := ./picture
OUTPUT_FILE = pic_$(DATE)

#FILE_OUTPUT = ./result/$(N)_$(INPUT_FILE)_$(GPU)
FILE_OUTPUT = stdout

GPU = --gpu
#PIC = --image-save $(OUTPUT_DIR)/$(OUTPUT_FILE)
#DEC = --decompose-save-firstGC dec_firstGC --decompose-save-firstGA dec_firstGA
ANALYS = --repeats-analysis-save $(OUTPUT_DIR)/$(OUTPUT_FILE)
#ANALYS = --matrix-gomology-save $(OUTPUT_DIR)/$(OUTPUT_FILE)
            #-F $(INPUT_DIR)/$(INPUT_FILE2)                              \

#USE_MATRIX = --use-matrix
            #-F $(INPUT_DIR)/$(INPUT_FILE2)                              \

ARGUMENTS = -f $(INPUT_DIR)/$(INPUT_FILE1)                              \
            --profiling-window     $(PROFILING_WINDOW)                  \
            --decompose-window     $(DECOMPOSE_WINDOW)                  \
            --step-decompose       $(STEP_DECOMPOSE)                    \
            --number-coefficient   $(NUMBER_COEFFICIENT)                \
            --eps                  $(EPS)                               \
            $(GPU) $(PIC) $(DEC) $(ANALYS) $(USE_MATRIX)


# HOST, MPI, LOMONOSOV or BLUEGENE
MACHINE := MPI

# short name NUMBER_PROC
N           = 4
NUMBER_PROC = $(N)

NODE        = 1
QUEUE       = test
TIME        = 10:00

#define mode compile
#USE_CUDA = 1
USE_MPI  = 1

#redefine compiler
ifdef USE_MPI
    CXX = mpicxx
endif

