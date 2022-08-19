#include "FieldTheoreticSimulation.h"

//----------------- Constructor ----------------------------
FieldTheoreticSimulation::FieldTheoreticSimulation(std::string str_platform, std::string model_name)
{
    this->factory = PlatformSelector::create_factory(str_platform);
    this->model_name = model_name;
}
FieldTheoreticSimulation::~FieldTheoreticSimulation()
{
    delete factory;
}
FieldTheoreticSimulation* FieldTheoreticSimulation::create_simulation(
        std::string str_platform, std::string model_name) 
{
    return new FieldTheoreticSimulation(str_platform,model_name);
}
std::string FieldTheoreticSimulation::get_model_name() 
{
    return model_name;
}
PolymerChain* FieldTheoreticSimulation::create_polymer_chain(
    std::vector<int> n_segment, 
    std::vector<double> bond_length) 
{
    return factory->create_polymer_chain(n_segment,bond_length,model_name);
}
SimulationBox* FieldTheoreticSimulation::create_simulation_box(
    std::vector<int> nx, std::vector<double> lx)
{
    return factory->create_simulation_box(nx,lx);
}
Pseudo* FieldTheoreticSimulation::create_pseudo(SimulationBox *sb, PolymerChain *pc)
{
    return factory->create_pseudo(sb,pc);
}
AndersonMixing* FieldTheoreticSimulation::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init) 
{
    return factory->create_anderson_mixing(n_var,max_hist,start_error,mix_min,mix_init);
}
void FieldTheoreticSimulation::display_info() 
{
    factory->display_info();
}