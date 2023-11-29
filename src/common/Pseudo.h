/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include <tuple>
#include <map>

#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "Exception.h"

class Pseudo
{
protected:
    ComputationBox *cb;
    Molecules *molecules;

    int n_complex_grid;

    void get_boltz_bond(double *boltz_bond, double bond_length_variance,
        std::vector<int> nx, std::vector<double> dx, double ds);

    void get_weighted_fourier_basis(
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::vector<int> nx, std::vector<double> dx);
public:
    Pseudo(ComputationBox *cb, Molecules *molecules);
    virtual ~Pseudo() {};

    int get_n_grid() {return cb->get_n_grid();};
    int get_n_blocks(int polymer) { Polymer& pc = molecules->get_polymer(polymer); return pc.get_n_blocks();};
    virtual void update_bond_function() = 0;
    // Inputs are in main memory
    virtual void compute_statistics(
        std::map<std::string, const double*> w_input,
        std::map<std::string, const double*> q_init) = 0;
    // Inputs are in platform memory (cpu or gpu)
    virtual void compute_statistics_device(
        std::map<std::string, const double*> d_w_input,
        std::map<std::string, const double*> d_q_init) = 0;
    virtual double get_total_partition(int polymer) = 0;
    virtual void get_total_concentration(std::string monomer_type, double *phi) = 0;
    virtual void get_total_concentration(int polymer, std::string monomer_type, double *phi) = 0;
    virtual void get_block_concentration(int polymer, double *phi) = 0;
    virtual std::vector<double> compute_stress() = 0;
    virtual void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) = 0;
};
#endif
