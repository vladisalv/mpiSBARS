# ============================================================================ #
# Version: 3.0                                                                 #
# Last update: 06.11.2014                                                      #
#                                                                              #
#                                                                              #
# !!!!!!!!!!!!!!!!!!!!!!! ВАЖНО !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#                                                                              #
#   При первом запуске выйдет ОШИБКА при линковке в окончательный модуль!      #
#   Это особенность данной версии Makefile. просто соберите проект еще раз!    #
#                                                                              #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#                                                                              #
#  если интересно в чем ошибка:                                                #
# дело в том, что откомпилированные объектные файлы сохраняются не корневой    #
# каталог, а в папку obj, но утилита make (видимо ее особенность) запоминает   #
# отсутствие файлов в корневом каталоге и потом ищет их только там (хотя опция #
# vpath %.o $(OBJ_NOW) стоит задолго до этого). В будущем исправлю             #
#                                                                              #
# Что нужно будет сделать:                                                     #
# 1. Исправить ошибку первого прохода                                          #
# ============================================================================ #

# structure:
# name in config
# name in Makefile (redifinition config)
# name in Makefile.sqel (redifinition Makefile, template)
# goals

# ====================  INCLUDE CONFIG FILE  ===================================

include $(wildcard config.mk)

# ========================  REDEFINE NAME  =====================================
PROGRAM_NAME ?= "PROGRAM"
VERSION_NUMBER ?= "UNKNOW"
# ----------------------  INPUT/OUTPUT FILES  ----------------------------------
INPUT_DIR   ?= input
INPUT_FILE  ?= input_$(DATE)
OUTPUT_DIR  ?= output
OUTPUT_FILE ?= output_$(DATE)
# ----------------------  LAUNCH OPTIONS  --------------------------------------
ARGUMENTS ?= -h
MACHINE ?= host
NODE ?=
NUMBER_PROC ?= 1
QUEUE ?=
TIME ?=
# ========================  DEFINE NAME  =======================================
BIN_NAME ?= $(PROGRAM_NAME)

TARGET ?= debug release
TARGET_NOW ?= debug
# -----------------------  CODE DIRECTORY  -------------------------------------
INCLUDE_DIR ?= include /opt/cuda/include
SRC_DIR ?= src
LIB_DIR ?= 
LIBRARY ?= 
#LIB_DIR ?= /usr/local/cuda/lib64/
#LIBRARY ?= cudart cublas
# ----------------------------  FLAGS  -----------------------------------------

# FLAGS := $(FLAGSCOMMON) $(FLAGSGOAL) $(FLAGSINCLUDES) $(FLAGLIBS) 
CUFLAGSGOAL = -arch=sm_20 -Xptxas -v -ccbin mpiCC

PRINT = @

# ====================  INCLUDE SKELETON FILE  =================================

include $(wildcard Makefile.skel)

# =============================  GOALS  ========================================

# абстрактные цели (выполняются в любом случае)
.PHONY: all run clean clean_exec clean_result test

# главная цель (пустая команда make)
all: build

build: mkdir $(OBJ_MODULES)
	echo Compiling program.....
	$(PRINT)$(CXX) $(CFLAGS) -L/opt/cuda/lib64 $(filter-out mkdir, $^) -o $(BIN_NOW)/$(BINARY_NAME) $(CFLAGSLIB) -lcudart

# запуск
run:
	$(PRINT)$(RUN) ./$(BIN_NOW)/$(BINARY_NAME) $(ARGUMENTS)
	#$(PRINT)$(RUN) ./$(BIN_NOW)/$(BINARY_NAME) $(ARGUMENTS) > $(FILE_OUTPUT)

rebuild: clean_exec build

clean: clean_exec clean_result

clean_exec:
	@echo clean exec
	$(PRINT)rm -f $(OBJ_NOW)/* $(BIN_NOW)/*

clean_result:
	@echo clean data
	$(PRINT)rm -f $(OUTPUT_DIR)/*

# вывести опции программы
option:
	$(PRINT)$(BIN_NOW)/$(BINARY_NAME) -h

# посмотреть свою очередь
watch:
	watch -n 1 squeue -u $(USER)

# отменить все поставленные задачи
cancel:
	scancel -u $(USER)

# создать все необходимые директории
mkdir:
	$(PRINT)mkdir -p $(BIN_NOW)
	$(PRINT)mkdir -p $(OBJ_NOW)
	$(PRINT)mkdir -p $(OUTPUT_DIR)

#test: $(OBJ_NOW)/options.o
test:
	#$(PRINT)$(CXX) $^ $(CFLAGS) test/opt.cpp  -o test/opt $(CFLAGSLIB)
	#./test/opt -h

begin:

finish:

# файл зависимостей
deps.mk: $(SRCMODULES)
	@echo "Create file of dependens..."
	@$(CC) $(CFLAGSINCLUDE) -MM $^ > $@
