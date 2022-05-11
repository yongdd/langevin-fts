/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include "SimulationBox.h"
#include "PolymerChain.h"

class Pseudo
{
protected:
    SimulationBox *sb;
    PolymerChain *pc;
    int n_complex_grid;

    void get_boltz_bond(double *boltz_bond, double bond_length_variance,
        std::array<int,3> nx, std::array<double,3> dx, double ds);

    void get_weighted_fourier_basis(
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::array<int,3> nx, std::array<double,3> dx);
public:
    Pseudo(SimulationBox *sb, PolymerChain *pc);
    virtual ~Pseudo() {};

    virtual void update() = 0;
    
    virtual void find_phi(
        double *phi_a,  double *phi_b,
        double *q1_init, double *q2_init,
        double *w_a, double *w_b, double &single_partition) = 0;
        
    virtual void get_partition(
        double *q1, int n1,
        double *q2, int n2) = 0;
        
    virtual std::array<double,3> dq_dl() = 0;

    // Methods for SWIG
    void find_phi(
        double **phi_a, int *len_p_a,
        double **phi_b, int *len_p_b,
        double *q1_init, int len_q1,
        double *q2_init, int len_q2,
        double *w_a, int len_w_a,
        double *w_b, int len_w_b,
        double &single_partition)
    {
        assert(len_q1  == sb->get_n_grid());
        assert(len_q2  == sb->get_n_grid());
        assert(len_w_a == sb->get_n_grid());
        assert(len_w_b == sb->get_n_grid());
        
        double *phi_a_alloc = (double *) malloc(sb->get_n_grid()*sizeof(double));
        double *phi_b_alloc = (double *) malloc(sb->get_n_grid()*sizeof(double));
        
        assert(phi_a_alloc != NULL);
        assert(phi_b_alloc != NULL);
        
        if (phi_a_alloc == NULL)
            std::cout << "Failed malloc() for phi_a" << std::endl;
        if (phi_b_alloc == NULL)
            std::cout << "Failed malloc() for phi_b" << std::endl;
            
        *phi_a = phi_a_alloc;
        *phi_b = phi_b_alloc;
        
        *len_p_a = sb->get_n_grid();
        *len_p_b = sb->get_n_grid();
        
        find_phi(*phi_a, *phi_b, q1_init, q2_init, w_a, w_b, single_partition);
    }
    void get_partition(
        double **q1_out, int *len_q1,  
        double **q2_out, int *len_q2, 
        int n1, int n2)
    {
        double *q1_out_alloc = (double *) malloc(sb->get_n_grid()*sizeof(double));
        double *q2_out_alloc = (double *) malloc(sb->get_n_grid()*sizeof(double));
        
        assert(q1_out_alloc != NULL);
        assert(q2_out_alloc != NULL);
        
        if (q1_out_alloc == NULL)
            std::cout << "Failed malloc() for q1_out" << std::endl;
        if (q2_out_alloc == NULL)
            std::cout << "Failed malloc() for q2_out" << std::endl;
            
        *q1_out = q1_out_alloc;
        *q2_out = q2_out_alloc;
        
        *len_q1 = sb->get_n_grid();
        *len_q2 = sb->get_n_grid();
        
        get_partition(*q1_out, n1, *q2_out, n2);
    }
};
#endif
