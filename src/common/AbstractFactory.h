/*----------------------------------------------------------
* class AbstractFactory
*-----------------------------------------------------------*/

#ifndef ABSTRACT_FACTORY_H_
#define ABSTRACT_FACTORY_H_

#include <string>
#include <array>
#include <complex>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h" 
#include "Array.h" 

// Design Pattern : Abstract Factory

template <typename T>
class AbstractFactory
{
protected:
    bool reduce_memory_usage;

public :
    virtual ~AbstractFactory() {};

    // virtual Array* create_array(
    //     unsigned int size) = 0;

    // virtual Array* create_array(
    //     double *data,
    //     unsigned int size) = 0;

    virtual ComputationBox<T>* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        const double* mask=nullptr) = 0;

    virtual Molecules* create_molecules_information(
        std::string chain_model, double ds, std::map<std::string, double> bond_lengths) = 0;

    PropagatorComputationOptimizer* create_propagator_computation_optimizer(Molecules* molecules, bool aggregate_propagator_computation)
    {
        return new PropagatorComputationOptimizer(molecules, aggregate_propagator_computation);
    };

    virtual PropagatorComputation<T>* create_pseudospectral_solver(
        ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer) = 0; 

    virtual PropagatorComputation<T>* create_realspace_solver(
        ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer) = 0; 

    virtual AndersonMixing<T>* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) = 0;

    // std::string get_model_name() {return chain_model;};
    virtual void display_info() = 0;
};
#endif
