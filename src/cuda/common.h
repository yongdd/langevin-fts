#ifndef __GPU_COMMON_H_
#define __GPU_COMMON_H_

#define TRUE 1
#define FALSE 0

#include <cufft.h>

#if USE_SINGLE_PRECISION == 1
typedef cufftReal    ftsReal;
typedef cufftComplex ftsComplex;

#else
typedef cufftDoubleReal    ftsReal;
typedef cufftDoubleComplex ftsComplex;
#endif

#endif
