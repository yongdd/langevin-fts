#include <iostream>
#include <cmath>
#include <complex>

#include "PropagatorComputation.h"

template <typename T>
PropagatorComputation<T>::PropagatorComputation(
    ComputationBox* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
{
    if (cb == nullptr)
        throw_with_line_number("ComputationBox* cb is a null pointer");
    if (molecules == nullptr)
        throw_with_line_number("Molecules *molecules is a null pointer");

    this->cb = cb;
    this->molecules = molecules;
    this->propagator_computation_optimizer = propagator_computation_optimizer;

    // Total partition functions for each polymer
    single_polymer_partitions = new T[molecules->get_n_polymer_types()];

    // Total partition functions for each solvent
    single_solvent_partitions = new T[molecules->get_n_solvent_types()];

    // Allocate memory for dq_dl
    for(int p=0; p<molecules->get_n_polymer_types(); p++){
        dq_dl.push_back({0.0, 0.0, 0.0});
    }
}
template <typename T>
PropagatorComputation<T>::~PropagatorComputation()
{
    delete[] single_polymer_partitions;
    delete[] single_solvent_partitions;
}
template <typename T>
std::vector<T> PropagatorComputation<T>::get_stress()
{ 
    const int DIM  = this->cb->get_dim();
    const int M    = this->cb->get_total_grid();
    std::vector<T> stress(DIM);

    int n_polymer_types = this->molecules->get_n_polymer_types();
    for(int d=0; d<DIM; d++)
        stress[d] = 0.0;
    
    for(int p=0; p<n_polymer_types; p++){
        Polymer& pc = this->molecules->get_polymer(p);
        for(int d=0; d<DIM; d++){
            stress[d] += this->dq_dl[p][d]*pc.get_volume_fraction()/pc.get_alpha()/this->single_polymer_partitions[p];
        }
    }
    return stress;
}
template <typename T>
std::vector<T> PropagatorComputation<T>::get_stress_gce(std::vector<double> fugacities)
{ 
    const int DIM  = this->cb->get_dim();
    const int M    = this->cb->get_total_grid();
    std::vector<T> stress(DIM);

    int n_polymer_types = this->molecules->get_n_polymer_types();
    for(int d=0; d<DIM; d++)
        stress[d] = 0.0;
    
    for(int p=0; p<n_polymer_types; p++){
        Polymer& pc = this->molecules->get_polymer(p);
        for(int d=0; d<DIM; d++){
            stress[d] += fugacities[p]*this->dq_dl[p][d];
        }
    }
    return stress;
}

// Explicit template instantiation for double and std::complex<double>
template class PropagatorComputation<double>;
template class PropagatorComputation<std::complex<double>>;