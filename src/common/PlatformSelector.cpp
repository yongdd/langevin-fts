
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
AbstractFactory *PlatformSelector::create_factory()
{
#ifdef USE_CPU_MKL
    return new MklFactory();
#endif
#ifdef USE_CUDA
    return new CudaFactory();
#endif
    throw_with_line_number("No available platform");
    return NULL;
}
AbstractFactory *PlatformSelector::create_factory(std::string str_platform)
{
#ifdef USE_CPU_MKL
    if (str_platform == "cpu-mkl")
        return new MklFactory();
#endif
#ifdef USE_CUDA
    if (str_platform == "cuda")
        return new CudaFactory();
#endif
    throw_with_line_number("Could not find platform '" + str_platform + "'");
    return NULL;
}

// try{
//     throw_with_line_number("Could not find platform '" + str_platform + "'");
// }
// catch(std::exception& exc)
// {
//     std::cerr << exc.what() << std::endl;
// }
