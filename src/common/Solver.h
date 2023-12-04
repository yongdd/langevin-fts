/*-------------------------------------------------------------
* This is an abstract Solver class
*------------------------------------------------------------*/

#ifndef SOLVER_H_
#define SOLVER_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include <tuple>
#include <map>

#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "PropagatorsAnalyzer.h"
#include "Exception.h"

class Solver
{
protected:
    ComputationBox *cb;
    Molecules *molecules;
    PropagatorsAnalyzer *propagators_analyzer;

    int n_complex_grid;

    void get_boltz_bond(double *boltz_bond, double bond_length_variance,
        std::vector<int> nx, std::vector<double> dx, double ds);

    void get_weighted_fourier_basis(
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::vector<int> nx, std::vector<double> dx);
public:
    Solver(ComputationBox *cb, Molecules *molecules, PropagatorsAnalyzer* propagators_analyzer);
    virtual ~Solver() {};

    int get_n_grid() {return cb->get_n_grid();};
    int get_n_blocks(int polymer) { Polymer& pc = molecules->get_polymer(polymer); return pc.get_n_blocks();};
    virtual void update_bond_function() = 0;
    // Inputs are in main memory
    virtual void compute_statistics(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {},
        double *q_mask = nullptr) = 0;
    // Inputs are in platform memory (cpu or gpu)
    virtual void compute_statistics_device(
        std::map<std::string, const double*> d_w_block,
        std::map<std::string, const double*> d_q_init = {},
        double *d_q_mask = nullptr) = 0;

    virtual double get_total_partition(int polymer) = 0;
    virtual void get_total_concentration(std::string monomer_type, double *phi) = 0;
    virtual void get_total_concentration(int polymer, std::string monomer_type, double *phi) = 0;
    virtual void get_block_concentration(int polymer, double *phi) = 0;
    virtual std::vector<double> compute_stress() = 0;
    virtual void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) = 0;

    virtual double get_solvent_partition(int s) = 0;
    virtual void get_solvent_concentration(int s, double *phi) = 0;

    // Check whether Q = int q(r,s)q^dagger(r,s) is constant w.r.t. variable s.
    virtual bool check_total_partition() = 0;
};
#endif
