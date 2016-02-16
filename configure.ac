#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([mpiSBARS], [0.8], [Vladislav.Toigildin@cs.msu.su])
AM_INIT_AUTOMAKE([mpiSBARS], [0.8])
AC_CONFIG_SRCDIR([src/main.cpp])
#AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
# AC_PROG_CXX do not use, cause we use mpic++

AC_ARG_WITH([mpi], [AS_HELP_STRING([--with-mpi],
    [compile with MPI (parallelization) support. If none is found,
    MPI is not used. Default: yes])
],,[with_mpi=yes])

if test x"${with_mpi}" = x"yes"; then
# check MPI compiler avaliable
    m4_include([m4/ax_prog_cxx_mpi.m4])
    AX_PROG_CXX_MPI([test x"${with_mpi}" != x"no"],
        [ AC_DEFINE(USE_MPI, 1, use MPI parallelization)],
        [AC_MSG_FAILURE([MPI compiler requested, but could not use MPI.])]
    )
fi
AM_CONDITIONAL([HAVE_USE_MPI], [test x"${with_mpi}" = x"yes"])


AC_ARG_ENABLE([gpu],
              [AS_HELP_STRING([--enable-gpu], [Enable feature using GPU device])],
              [enable_gpu=yes],
              [enable_gpu=no]
             )

if test x"${enable_gpu}" = x"yes"; then
# TODO: we need check nvcc
    AC_CHECK_HEADERS([cuda_runtime.h], , AC_MSG_ERROR(cannot find cuda_runtime.h))
    AC_CHECK_LIB([cudart], [main], , [AC_MSG_ERROR(cannot find cudart)])
    AC_DEFINE(USE_CUDA, 1, use CUDA massage)
fi
AM_CONDITIONAL([HAVE_USE_CUDA], [test x"${enable_gpu}" = x"yes"])

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


AC_OUTPUT(Makefile src/Makefile)
