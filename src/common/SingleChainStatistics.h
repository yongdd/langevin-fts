/*----------------------------------------------------------
* class SingleChainStatistics
*-----------------------------------------------------------*/

#ifndef SINGLE_CHAIN_STATISTICS_H_
#define SINGLE_CHAIN_STATISTICS_H_

#include <string>

#include "PlatformSelector.h"
#include "AbstractFactory.h"

class SingleChainStatistics
{
private:
    AbstractFactory* factory;
    std::string model_name;

public:
    SingleChainStatistics(std::string str_platform, std::string model_name);
    ~SingleChainStatistics();
    static SingleChainStatistics* create_computation(
        std::string str_platform, std::string model_name);
    std::string get_model_name();
    PolymerChain* create_polymer_chain(
        std::vector<int> n_segment, 
        std::vector<double> bond_length);
    ComputationBox* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx);
    Pseudo* create_pseudo(ComputationBox *cb, PolymerChain *pc);
    AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init);
    void display_info();
};

#endif