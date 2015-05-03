#ifndef __TYPES_HEADER__
#define __TYPES_HEADER__

typedef unsigned int uint;
typedef unsigned long ulong;

typedef char   TypeSequence;
typedef uint   TypeProfile;
typedef float  TypeDecomposition;
typedef bool   TypeGomology;

#define MPI_TFLOAT MPI_FLOAT

#ifdef USE_MPI
    #define MPI_TYPE_SEQUENCE      MPI_CHAR
    #define MPI_TYPE_PROFILE       MPI_UNSIGNED_LONG
    #define MPI_TYPE_DECOMPOSITION MPI_FLOAT
    #define MPI_TYPE_GOMOLOGY      MPI::BOOL
    #define MPI_TYPE_LIST_REPEAT   MPI_UNSIGNED_LONG
#else
    typedef int MPI_Datatype
    #define MPI_TYPE_SEQUENCE      0
    #define MPI_TYPE_PROFILE       0
    #define MPI_TYPE_DECOMPOSITION 0
    #define MPI_TYPE_GOMOLOGY      0
    #define MPI_TYPE_LIST_REPEAT   0
#endif

#ifdef DEBUG_MODE
    #define DEBUG(x) x
#else
    #define DEBUG(x) ;
#endif

#ifndef M_PI
    #define M_PI 3.1415926535897932384626433832795
#endif

#define SQRT_2 1.4142135623730950488016887242097

#endif /* __TYPES_HEADER__ */
