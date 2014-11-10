#ifndef __TYPES_HEADER__
#define __TYPES_HEADER__

typedef unsigned int uint;
typedef unsigned long ulong;

typedef char   TypeSequence;
typedef uint   TypeProfile;
typedef double TypeDecomposition;
typedef bool   TypeGomology;
typedef bool   TypeAnalysis;

#define MPI_TFLOAT MPI_DOUBLE

#ifdef DEBUG_MODE
    #define DEBUG(x) x
#else
    #define DEBUG(x) ;
#endif

#define M_PI 3.1415926535897932384626433832795
#define SQRT_2 1.4142135623730950488016887242097

#endif /* __TYPES_HEADER__ */
