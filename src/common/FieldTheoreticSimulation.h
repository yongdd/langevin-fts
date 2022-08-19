/*----------------------------------------------------------
* class FieldTheoreticSimulation
*-----------------------------------------------------------*/

#ifndef Field_Simulation_H_
#define Field_Simulation_H_

#include <string>

#include "PlatformSelector.h"
#include "AbstractFactory.h"

class FieldTheoreticSimulation
{
private:
    AbstractFactory* factory;
    std::string model_name;

public:
    FieldTheoreticSimulation(std::string str_platform, std::string model_name);
    ~FieldTheoreticSimulation();
    static FieldTheoreticSimulation* create_simulation(
        std::string str_platform, std::string model_name);
    std::string get_model_name();
    PolymerChain* create_polymer_chain(
        std::vector<int> n_segment, 
        std::vector<double> bond_length);
    SimulationBox* create_simulation_box(
        std::vector<int> nx,
        std::vector<double> lx);
    Pseudo* create_pseudo(SimulationBox *sb, PolymerChain *pc);
    AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init);
    void display_info();
};

#endif