/*-------------------------------------------------------------
* This is an abstract PropagatorComputation class
*------------------------------------------------------------*/

#ifndef PROPAGATOR_COMPUTATION_H_
#define PROPAGATOR_COMPUTATION_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include <tuple>
#include <map>

#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"
#include "Exception.h"

template <typename T>
class PropagatorComputation
{
protected:
    ComputationBox<T>* cb;
    Molecules *molecules;
    PropagatorComputationOptimizer *propagator_computation_optimizer;

    // Total partition functions for each polymer
    T* single_polymer_partitions;

    // Total partition functions for each solvent
    T* single_solvent_partitions;

    // Stress of each polymer
    std::vector<std::array<T,3>> dq_dl;
public:
    PropagatorComputation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer);
    virtual ~PropagatorComputation();

    int get_total_grid() {return this->cb->get_total_grid();};
    int get_n_blocks(int polymer) { Polymer& pc = this->molecules->get_polymer(polymer); return pc.get_n_blocks();};
    virtual void update_laplacian_operator() = 0;

    virtual void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) = 0;

    virtual void compute_concentrations() = 0;

    virtual void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) = 0;

    virtual void compute_stress() = 0;
    virtual T get_total_partition(int polymer) = 0;
    virtual void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) = 0;

    // Canonical ensemble
    virtual void get_total_concentration(std::string monomer_type, T *phi) = 0;
    virtual void get_total_concentration(int polymer, std::string monomer_type, T *phi) = 0;
    virtual void get_block_concentration(int polymer, T *phi) = 0;

    virtual T get_solvent_partition(int s) = 0;
    virtual void get_solvent_concentration(int s, T *phi) = 0;

    virtual std::vector<T> get_stress();

    // Grand canonical ensemble
    virtual void get_total_concentration_gce  (double fugacity, int polymer, std::string monomer_type, T *phi) = 0;
    virtual std::vector<T> get_stress_gce(std::vector<double> fugacities);

    // Check whether Q = int q(r,s)q^dagger(r,s) is constant w.r.t. variable s.
    virtual bool check_total_partition() = 0;

};
#endif
