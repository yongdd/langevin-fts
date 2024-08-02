#include <iostream>
#include <cmath>
#include "PropagatorComputation.h"

PropagatorComputation::PropagatorComputation(
    ComputationBox *cb,
    Molecules *molecules,
    PropagatorAnalyzer *propagator_analyzer)
{
    if (cb == nullptr)
        throw_with_line_number("ComputationBox *cb is a null pointer");
    if (molecules == nullptr)
        throw_with_line_number("Molecules *molecules is a null pointer");

    this->cb = cb;
    this->molecules = molecules;
    this->propagator_analyzer = propagator_analyzer;

    // Total partition functions for each polymer
    single_polymer_partitions = new double[molecules->get_n_polymer_types()];

    // Total partition functions for each solvent
    single_solvent_partitions = new double[molecules->get_n_solvent_types()];

    // Allocate memory for dq_dl
    for(int p=0; p<molecules->get_n_polymer_types(); p++)
        dq_dl.push_back({0.0, 0.0, 0.0,});
}
PropagatorComputation::~PropagatorComputation()
{
    delete[] single_polymer_partitions;
    delete[] single_solvent_partitions;
}

std::vector<double> PropagatorComputation::get_stress()
{ 
    const int DIM  = cb->get_dim();
    std::vector<double> stress(DIM);
    const int M    = cb->get_n_grid();

    int n_polymer_types = molecules->get_n_polymer_types();
    for(int d=0; d<DIM; d++)
        stress[d] = 0.0;
    
    for(int p=0; p<n_polymer_types; p++){
        Polymer& pc = molecules->get_polymer(p);
        for(int d=0; d<DIM; d++){
            stress[d] += dq_dl[p][d]*pc.get_volume_fraction()/pc.get_alpha()/single_polymer_partitions[p];
        }
    }
    return stress;
}
std::vector<double> PropagatorComputation::get_stress_gce(std::vector<double> fugacities)
{ 
    const int DIM  = cb->get_dim();
    std::vector<double> stress(DIM);
    const int M    = cb->get_n_grid();

    int n_polymer_types = molecules->get_n_polymer_types();
    for(int d=0; d<DIM; d++)
        stress[d] = 0.0;
    
    for(int p=0; p<n_polymer_types; p++){
        Polymer& pc = molecules->get_polymer(p);
        for(int d=0; d<DIM; d++){
            stress[d] += fugacities[p]*dq_dl[p][d];
        }
    }
    return stress;
}