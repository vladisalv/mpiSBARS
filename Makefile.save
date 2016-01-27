# ============================================================================ #
# structure:                                                                   #
# name in config                                                               #
# name in Makefile (redifinition config)                                       #
# name in Makefile.sqel (redifinition Makefile, template)                      #
# goals                                                                        #
#                                                                              #
# ============================================================================ #


# ====================  INCLUDE CONFIG FILE  ===================================

include $(wildcard config.mk)

# ========================  REDEFINE NAME  =====================================
PROGRAM_NAME ?= "PROGRAM"
VERSION ?= $(TARGET_NOW)
VERSION_NUMBER ?= "UNKNOW"
TARGET ?= debug release
TARGET_NOW ?= debug
# ----------------------  INPUT/OUTPUT FILES  ----------------------------------
INPUT_DIR   ?= test/input
INPUT_FILE  ?= input_$(DATE)
OUTPUT_DIR  ?= test/output
OUTPUT_FILE ?= output_$(DATE)
# ----------------------  LAUNCH OPTIONS  --------------------------------------
ARGUMENTS ?= -h
MACHINE ?= MPI
NUMBER_NODE?= 1
NUMBER_PROC ?= 1
NODE_TASK ?= 1
QUEUE ?= gputest
TIME ?= 15:00
# -----------------------  CODE DIRECTORY  -------------------------------------
INC_DIR ?= include/ /opt/cuda/include /opt/cuda/cuda-6.5/include/
SRC_DIR ?= src/
LIB_DIR ?= /usr/local/cuda/lib64/ /opt/cuda/cuda-6.5/lib64/
ifdef USE_CUDA
    LIBRARY ?= cudart cublas
endif
# ----------------------------  FLAGS  -----------------------------------------
CUFLAGSGOAL = -arch=sm_20 -Xptxas -v
ifdef USE_MPI
    CUFLAGSGOAL += -ccbin mpiCC
endif

PRINT = @
# ====================  INCLUDE SKELETON FILE  =================================

include $(wildcard Makefile.skel)

# =============================  GOALS  ========================================

# абстрактные цели (выполняются в любом случае)
.PHONY: run clean clean_exec clean_result test

# главная цель (пустая команда make)
all: build

build: mkdir $(OBJ_MODULES)
	$(PRINT)echo Compiling program.....
	$(PRINT)$(CXX) $(CXXFLAGS) $(addprefix $(OBJ_NOW), $(notdir $(filter-out mkdir, $^))) -o $(BIN_NOW)/$(BINARY_NAME) $(CXXFLAGSLIBRARY)

# запуск
run:
	$(PRINT)$(RUN) ./$(BIN_NOW)/$(BINARY_NAME) $(ARGUMENTS)

rebuild: clean_exec build

clean: clean_exec clean_result

clean_exec:
	$(PRINT)echo clean exec
	$(PRINT)rm -f $(OBJ_NOW)/* $(BIN_NOW)/*

clean_result:
	$(PRINT)echo clean data
	$(PRINT)rm -f $(OUTPUT_DIR)/*

# вывести опции программы
option:
	$(PRINT)$(BIN_NOW)/$(BINARY_NAME) -h

# посмотреть свою очередь
watch:
	watch -n 1 $(WATCH) -u $(USER)

# отменить все поставленные задачи
cancel:
	$(CANCEL) -u $(USER)

# создать все необходимые директории
mkdir:
	$(PRINT)mkdir -p $(BIN_NOW)
	$(PRINT)mkdir -p $(OBJ_NOW)
	$(PRINT)mkdir -p $(OUTPUT_DIR)

test:
	#$(PRINT)$(CXX) $^ $(CFLAGS) test/opt.cpp  -o test/opt $(CFLAGSLIB)
	#./test/opt -h

begin:

finish:

# файл зависимостей
deps.mk: $(SRCMODULES)
	@echo "Create file of dependens..."
	@$(CC) $(CFLAGSINCLUDE) -MM $^ > $@
