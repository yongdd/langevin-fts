
#include <iostream>
#include <vector>
#include <string>
#include "Exception.h"

#ifdef USE_CPU_MKL
#include "MklFactory.h"
#endif
#ifdef USE_CUDA
#include "CudaFactory.h"
#include "CudaCommon.h"
#endif
#include "PlatformSelector.h"

std::vector<std::string> PlatformSelector::avail_platforms()
{
    std::vector<std::string> names;
#ifdef USE_CPU_MKL
    names.push_back("cpu-mkl");
#endif
#ifdef USE_CUDA
    names.push_back("cuda");
#endif
    if(names.size() == 0)
        throw_with_line_number("No available platform");
    return names;
}
AbstractFactory *PlatformSelector::create_factory(std::string platform)
{
#ifdef USE_CPU_MKL
    if (platform == "cpu-mkl")
        return new MklFactory("double", false);
#endif
#ifdef USE_CUDA
    if (platform == "cuda")
        return new CudaFactory("double", false);
#endif
    throw_with_line_number("Could not find platform '" + platform + "'");
    return NULL;
}
AbstractFactory *PlatformSelector::create_factory(std::string platform, bool reduce_memory_usage)
{
#ifdef USE_CPU_MKL
    if (platform == "cpu-mkl")
        return new MklFactory("double", reduce_memory_usage);
#endif
#ifdef USE_CUDA
    if (platform == "cuda")
        return new CudaFactory("double", reduce_memory_usage);
#endif
    throw_with_line_number("Could not find platform '" + platform + "'");
    return NULL;
}
AbstractFactory *PlatformSelector::create_factory(std::string platform, std::string data_type)
{
#ifdef USE_CPU_MKL
    if (platform == "cpu-mkl")
        return new MklFactory(data_type, false);
#endif
#ifdef USE_CUDA
    if (platform == "cuda")
        return new CudaFactory(data_type, false);
#endif
    throw_with_line_number("Could not find platform '" + platform + "'");
    return NULL;
}
AbstractFactory *PlatformSelector::create_factory(std::string platform, std::string data_type, bool reduce_memory_usage)
{
#ifdef USE_CPU_MKL
    if (platform == "cpu-mkl")
        return new MklFactory(data_type, reduce_memory_usage);
#endif
#ifdef USE_CUDA
    if (platform == "cuda")
        return new CudaFactory(data_type, reduce_memory_usage);
#endif
    throw_with_line_number("Could not find platform '" + platform + "'");
    return NULL;
}