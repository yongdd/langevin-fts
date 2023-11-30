/*----------------------------------------------------------
* class AbstractFactory
*-----------------------------------------------------------*/

#ifndef ABSTRACT_FACTORY_H_
#define ABSTRACT_FACTORY_H_

#include <string>
#include <array>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "Propagators.h"
#include "Solver.h"
#include "AndersonMixing.h" 
#include "Array.h" 

// Design Pattern : Abstract Factory

class AbstractFactory
{
protected:
    std::string chain_model;
    bool reduce_memory_usage;
public :
    virtual ~AbstractFactory() {};

    virtual Array* create_array(
        unsigned int size) = 0;

    virtual Array* create_array(
        double *data,
        unsigned int size) = 0;

    virtual ComputationBox* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx) = 0;

    virtual Molecules* create_molecules_information(
        std::string chain_model, double ds, std::map<std::string, double> bond_lengths) = 0;

    Propagators* create_propagators_information(Molecules* molecules, bool aggregate_propagator_computation)
    {
        return new Propagators(molecules, aggregate_propagator_computation);
    };

    virtual Solver* create_pseudospectral_solver(
        ComputationBox *cb, Molecules *molecules, Propagators* propagators) = 0; 

    virtual AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) = 0;

    std::string get_model_name() {return chain_model;};
    virtual void display_info() = 0;
};
#endif
