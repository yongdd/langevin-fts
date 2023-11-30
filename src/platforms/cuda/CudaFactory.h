/*----------------------------------------------------------
* class CudaFactory
*-----------------------------------------------------------*/

#ifndef CUDA_FACTORY_H_
#define CUDA_FACTORY_H_

#include "ComputationBox.h"
#include "Polymer.h"
#include "Solver.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "Array.h"

class CudaFactory : public AbstractFactory
{
public :
    CudaFactory(bool reduce_memory_usage);

    Array* create_array(
        unsigned int size) override;

    Array* create_array(
        double *data,
        unsigned int size) override;

    ComputationBox* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx) override;

    Molecules* create_molecules_information(
        std::string chain_model, double ds, std::map<std::string, double> bond_lengths) override;

    Solver* create_pseudospectral_solver(ComputationBox *cb, Molecules *molecules, Propagators* propagators) override;

    AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;

    void display_info() override;
};
#endif
