#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([mpiSBARS], [0.8], [Vladislav.Toigildin@cs.msu.su])
AM_INIT_AUTOMAKE([mpiSBARS], [0.8])
AC_CONFIG_SRCDIR([src/main.cpp])
#AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.
#AC_HAVE_LIBRARY(cudart, AC_DEFINE(HAVE_CUDART, 1, have cudart library))
#AC_HAVE_LIBRARY(cublas, AC_DEFINE(HAVE_CUBLAS, 1, have cublas library))
#AC_HAVE_LIBRARY(cudart)
#AC_HAVE_LIBRARY(cublas)

AC_ARG_ENABLE([gpu],
              [AS_HELP_STRING([--enable-gpu], [Enable feature using GPU device])],
              [enable_gpu=yes],
              [enable_gpu=no])

if test x"${enable_gpu}" = x"yes"; then
    AC_CHECK_HEADERS([cuda_runtime.h], , AC_MSG_ERROR(cannot find cuda_runtime.h))
    AC_CHECK_LIB([cudart], [main], , [AC_MSG_ERROR(cannot find cudart)])
# we need check nvcc
# install nvcc option
    AC_DEFINE(USE_CUDA, 1, use CUDA massage)
fi

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h ])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([memset sqrt])

AC_DEFINE(USE_MPI,  1, use MPI massage)

AM_CONDITIONAL([HAVE_USE_CUDA], [test x"${enable_gpu}" = x"yes"])

AC_OUTPUT(Makefile src/Makefile)
