#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "SimulationBox.h"
#include "CpuPseudo.h"
#include "MklFFT.h"
#include "FftwFFT.h"
#include "CpuAndersonMixing.h"

#include "CudaSimulationBox.h"
#include "CudaPseudo.h"
#include "CudaAndersonMixing.h"

#include "KernelFactory.h"

KernelFactory::KernelFactory()
{
    std::vector<std::string> valid_strings;

#ifdef USE_CPU_MKL
    valid_strings.push_back("CPU_MKL");
#endif
#ifdef USE_CPU_FFTW
    valid_strings.push_back("CPU_FFTW");
#endif
#ifdef USE_CUDA
    valid_strings.push_back("CUDA");
#endif
    this->str_platform = valid_strings.back();
}

KernelFactory::KernelFactory(std::string str_platform)
{
    std::vector<std::string> valid_strings;

#ifdef USE_CUDA
    valid_strings.push_back("CUDA");
#endif
#ifdef USE_CPU_MKL
    valid_strings.push_back("CPU_MKL");
#endif
#ifdef USE_CPU_FFTW
    valid_strings.push_back("CPU_FFTW");
#endif

    this->str_platform = str_platform;
    
    bool valid = false;
    for(unsigned int i=0; i<valid_strings.size(); i++){
        valid = valid || (str_platform == valid_strings[i]);
    }
    
    if(!valid)
    {
        std::cerr<< "Invalid platform type: " << str_platform << std::endl;
        exit(-1);
    }
}
PolymerChain* KernelFactory::create_polymer_chain(double f, int NN, double chi_n)
{
    return new PolymerChain(f, NN, chi_n);
}
SimulationBox* KernelFactory::create_simulation_box(
    std::array<int,3> nx, std::array<double,3>  lx)
{
#ifdef USE_CPU_MKL
    if (str_platform == "CPU_MKL")
        return new SimulationBox(nx, lx);
#endif
#ifdef USE_CPU_FFTW
    if( str_platform == "CPU_FFTW")
        return new SimulationBox(nx, lx);
#endif
#ifdef USE_CUDA
    if (str_platform == "CUDA")
        return new CudaSimulationBox(nx, lx);
    return NULL;
#endif
}
Pseudo* KernelFactory::create_pseudo(SimulationBox *sb, PolymerChain *pc)
{
#ifdef USE_CPU_MKL
    if (str_platform == "CPU_MKL")
        return new CpuPseudo(sb, pc, new MklFFT(sb->get_nx()));
#endif
#ifdef USE_CPU_FFTW
    if (str_platform == "CPU_FFTW")
        return new CpuPseudo(sb, pc, new FftwFFT(sb->get_nx()));
#endif
#ifdef USE_CUDA
    if (str_platform == "CUDA")
        return new CudaPseudo(sb, pc);
#endif
    return NULL;
}
AndersonMixing* KernelFactory::create_anderson_mixing(
    SimulationBox *sb, int n_comp,
    double max_anderson, double start_anderson_error,
    double mix_min, double mix_init)
{
#ifdef USE_CPU_MKL
    if (str_platform == "CPU_MKL")
        return new CpuAndersonMixing(
                   sb, n_comp, max_anderson,
                   start_anderson_error, mix_min, mix_init);
#endif
#ifdef USE_CPU_FFTW
    if (str_platform == "CPU_FFTW")
        return new CpuAndersonMixing(
                   sb, n_comp, max_anderson,
                   start_anderson_error, mix_min, mix_init);
#endif
#ifdef USE_CUDA
    if (str_platform == "CUDA")
        return new CudaAndersonMixing(
                   sb, n_comp, max_anderson,
                   start_anderson_error, mix_min, mix_init);
#endif
    return NULL;
}
