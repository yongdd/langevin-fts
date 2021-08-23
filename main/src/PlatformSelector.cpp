
#include <iostream>
#include <vector>
#include <string>
#include "MklFactory.h"
#include "FftwFactory.h"
#include "CudaFactory.h"
#include "PlatformSelector.h"

AbstractFactory* PlatformSelector::create_factory(std::string str_platform)
{
#ifdef USE_CPU_MKL
    if (str_platform == "CPU_MKL")
        return new MklFactory();
#endif
#ifdef USE_CPU_FFTW
    if (str_platform == "CPU_FFTW")
        return new FftwFactory();
#endif
#ifdef USE_CUDA
    if (str_platform == "CUDA")
        return new CudaFactory();
#endif
    return NULL;
}
