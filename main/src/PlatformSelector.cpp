
#include <iostream>
#include <vector>
#include <string>
#include "MklFactory.h"
#include "FftwFactory.h"
#include "CudaFactory.h"
#include "PlatformSelector.h"

PlatformSelector::PlatformSelector()
{
    init_valid_strings();
    this->str_platform = valid_strings.back();
}
PlatformSelector::PlatformSelector(std::string str_platform)
{
    init_valid_strings();
    this->str_platform = str_platform;
    bool valid = false;
    for(unsigned int i=0; i<valid_strings.size(); i++)
    {
        valid = valid || (str_platform == valid_strings[i]);
    }

    if(!valid)
    {
        std::cerr<< "Invalid platform type: " << str_platform << std::endl;
        exit(-1);
    }
}
void PlatformSelector::init_valid_strings()
{
#ifdef USE_CPU_MKL
    valid_strings.push_back("CPU_MKL");
#endif
#ifdef USE_CPU_FFTW
    valid_strings.push_back("CPU_FFTW");
#endif
#ifdef USE_CUDA
    valid_strings.push_back("CUDA");
#endif
}
AbstractFactory* PlatformSelector::create_factory()
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
