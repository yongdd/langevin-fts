/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_BRANCHED_H_
#define PSEUDO_BRANCHED_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include <tuple>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Exception.h"

namespace py = pybind11;

struct branched_pseudo_edge{
    int max_n_segment;                              // the maximum segment number
    double*  partition;                             // array for partition function
    std::string species;                            // species
    std::vector<std::pair<std::string, int>> deps;  // dependency pairs
};

struct branched_pseudo_block{
    int n_segment;               // segment number
    std::string species;         // species
    double* phi;                 // array for concentration
};

class Pseudo
{
protected:
    ComputationBox *cb;
    PolymerChain *pc;
    int n_complex_grid;

    void get_boltz_bond(double *boltz_bond, double bond_length_variance,
        std::array<int,3> nx, std::array<double,3> dx, double ds);

    void get_weighted_fourier_basis(
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::array<int,3> nx, std::array<double,3> dx);
public:
    Pseudo(ComputationBox *cb, PolymerChain *pc);
    virtual ~Pseudo() {};

    virtual void update() = 0;
    
    virtual void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        double *phi, double &single_partition) = 0;
    virtual std::array<double,3> dq_dl() = 0;
    virtual void get_partition(double *q_out, int v, int u, int n) = 0;

    // Methods for pybind11
    std::tuple<py::array_t<double>, double>
    compute_statistics(std::map<std::string,py::array_t<double>> q_init, std::map<std::string,py::array_t<double>> w_block)
    {
        const int M = cb->get_n_grid();
        const int N_B = pc->get_n_block();

        std::map<std::string,double*> map_buf_q_init;
        std::map<std::string,double*> map_buf_w_block;

        for (auto it=q_init.begin(); it!=q_init.end(); ++it)
        {
            //buf_q_init
            py::buffer_info buf_q_init = it->second.request();
            if (buf_q_init.shape[0] != M){
                throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_q_init.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
            }
            else
            {
                map_buf_q_init.insert(std::pair<std::string,double*>(it->first,(double *)buf_q_init.ptr));
            }
        }

        for (auto it=w_block.begin(); it!=w_block.end(); ++it)
        {
            //buf_w_block
            py::buffer_info buf_w_block = it->second.request();
            if (buf_w_block.shape[0] != M){
                throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_w_block.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
            }
            else
            {
                map_buf_w_block.insert(std::pair<std::string,double*>(it->first,(double *)buf_w_block.ptr));
            }
        }
        
        try{
            double single_partition;
            py::array_t<double> phi = py::array_t<double>({N_B,M});
            py::buffer_info buf_phi = phi.request();
            
            compute_statistics(
                map_buf_q_init, map_buf_w_block,
                (double*) buf_phi.ptr, single_partition);
            return std::make_tuple(std::move(phi), single_partition);
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    };
    py::array_t<double> get_partition(int v, int u, int n)
    {
        try{
            const int M = cb->get_n_grid();
            py::array_t<double> q1 = py::array_t<double>(M);
            py::buffer_info buf_q1 = q1.request();

            get_partition((double*) buf_q1.ptr, v, u, n);
            
            return std::move(q1);
        }
        catch(std::exception& exc)
        {
            throw_with_line_number(exc.what());
        }
    };
};
#endif
