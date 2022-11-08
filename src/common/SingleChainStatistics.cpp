#include "SingleChainStatistics.h"

//----------------- Constructor ----------------------------
SingleChainStatistics::SingleChainStatistics(std::string str_platform, std::string model_name)
{
    this->factory = PlatformSelector::create_factory(str_platform);
    this->model_name = model_name;
}
SingleChainStatistics::~SingleChainStatistics()
{
    delete factory;
}
SingleChainStatistics* SingleChainStatistics::create_computation(
        std::string str_platform, std::string model_name) 
{
    return new SingleChainStatistics(str_platform, model_name);
}
std::string SingleChainStatistics::get_model_name() 
{
    return model_name;
}
PolymerChain* SingleChainStatistics::create_polymer_chain(
    std::vector<int> n_segment, 
    std::vector<double> bond_length) 
{
    return factory->create_polymer_chain(n_segment,bond_length,model_name);
}
ComputationBox* SingleChainStatistics::create_computation_box(
    std::vector<int> nx, std::vector<double> lx)
{
    return factory->create_computation_box(nx,lx);
}
Pseudo* SingleChainStatistics::create_pseudo(ComputationBox *cb, PolymerChain *pc)
{
    return factory->create_pseudo(cb,pc);
}
AndersonMixing* SingleChainStatistics::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init) 
{
    return factory->create_anderson_mixing(n_var,max_hist,start_error,mix_min,mix_init);
}
void SingleChainStatistics::display_info() 
{
    factory->display_info();
}