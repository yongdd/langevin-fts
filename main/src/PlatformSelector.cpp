
#include <iostream>
#include <vector>
#include <string>
#include "MklFactory.h"
#include "FftwFactory.h"
#include "CudaFactory.h"
#include "CudaCommon.h"
#include "PlatformSelector.h"

std::vector<std::string> PlatformSelector::avail_platforms()
{
    std::vector<std::string> names;
#ifdef USE_CPU_MKL
    names.push_back("cpu-mkl");
#endif
#ifdef USE_CPU_FFTW
    names.push_back("cpu-fftw");
#endif
#ifdef USE_CUDA
    names.push_back("cuda");
#endif
    return names;
}
AbstractFactory* PlatformSelector::create_factory(std::string str_platform)
{
#ifdef USE_CPU_MKL
    if (str_platform == "cpu-mkl")
        return new MklFactory();
#endif
#ifdef USE_CPU_FFTW
    if (str_platform == "cpu-fftw")
        return new FftwFactory();
#endif
#ifdef USE_CUDA
    if (str_platform == "cuda")
        return new CudaFactory();
#endif
    std::cerr << "Could not find : '" << str_platform << "'" << std::endl;
    return NULL;
}
