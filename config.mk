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
#INPUT_FILE := file5M
#INPUT_FILE := file10M
INPUT_FILE := file50K

OUTPUT_DIR := ./picture
OUTPUT_FILE = pic_$(DATE)

GPU = --gpu
PIC = --output $(OUTPUT_DIR)/$(OUTPUT_FILE)
DEC = --decompose-save-firstGC dec_firstGC --decompose-save-firstGA dec_firstGA
ANALYS = --matrix-analysis-save $(OUTPUT_DIR)/$(OUTPUT_FILE)

ARGUMENTS = -f $(INPUT_DIR)/$(INPUT_FILE)                              \
            --profiling-window     $(PROFILING_WINDOW)                  \
            --decompose-window     $(DECOMPOSE_WINDOW)                  \
            --step-decompose       $(STEP_DECOMPOSE)                    \
            --number-coefficient   $(NUMBER_COEFFICIENT)                \
            --eps                  $(EPS)                               \
            $(GPU) $(PIC) $(DEC) $(ANALYS)


# HOST, MPI, LOMONOSOV or BLUEGENE
MACHINE := MPI

# short name NUMBER_PROC
N           = 4
NUMBER_PROC = $(N)

NODE        = 1
QUEUE       = test
TIME        = 10:00

#redefine compiler
CXX = mpicxx
