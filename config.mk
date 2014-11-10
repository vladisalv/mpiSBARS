PROGRAM_NAME  := SSSDNA
VERSION_NUMER := 3.0
DEBUG_MODE    := 1

PROFILING_WINDOW   = 250
DECOMPOSE_WINDOW   = 250
STEP_DECOMPOSE     = 100
NUMBER_COEFFICIENT = 75
EPS                = 0.01
INPUT_DIR := ./test/samples
INPUT_FILE := file50K

OUTPUT_DIR := ./picture
OUTPUT_FILE = pic_$(DATE)

ARGUMENTS := -f $(INPUT_DIR)/$(INPUT_FILE) -o $(OUTPUT_DIR)/$(OUTPUT_FILE) \
             --profiling-window   $(PROFILING_WINDOW)   \
             --decompose-window   $(DECOMPOSE_WINDOW)   \
             --step-decompose     $(STEP_DECOMPOSE)     \
             --number-coefficient $(NUMBER_COEFFICIENT) \
             --eps                $(EPS)

# HOST, MPI, LOMONOSOV or BLUEGENE
MACHINE := MPI

# short name NUMBER_PROC
N           = 4
NUMBER_PROC = $(N)

NODE        = 1
QUEUE       =
TIME        =

#redefine compiler
CXX = mpicxx
