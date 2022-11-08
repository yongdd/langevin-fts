/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Exception.h"

namespace py = pybind11;

class Pseudo
{
protected:
    ComputationBox *cb;
    PolymerChain *pc;
    int n_complex_grid;
    int n_block;

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
        double *phi, double *q_1_init, double *q_2_init,
        double *w_block, double &single_partition) = 0;
        
    virtual void get_partition(
        double *q1, int n1,
        double *q2, int n2) = 0;
        
    virtual std::array<double,3> dq_dl() = 0;

    // Methods for pybind11
    std::tuple<py::array_t<double>, double>
    compute_statistics(py::array_t<double> q1_init, py::array_t<double> q2_init, py::array_t<double> w_block)
    {
        const int M = cb->get_n_grid();
        const int N_B = pc->get_n_block();
        py::buffer_info buf_q1_init = q1_init.request();
        py::buffer_info buf_q2_init = q2_init.request();
        py::buffer_info buf_w_block = w_block.request();
        if (buf_q1_init.size != M)
            throw_with_line_number("Size of input q1_init (" + std::to_string(buf_q1_init.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
        if (buf_q2_init.size != M)
            throw_with_line_number("Size of input q2_init (" + std::to_string(buf_q2_init.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
        if (buf_w_block.ndim == 1)
        {
            if (buf_w_block.shape[0] != M)
                throw_with_line_number("Size of input w ("       + std::to_string(buf_w_block.size) + ") and 'n_block*n_grid' (" + std::to_string(M) + ") must match");
        }
        else if (buf_w_block.ndim == 2)
        {
            if (buf_w_block.size != N_B*M)
                throw_with_line_number("Size of input w ("       + std::to_string(buf_w_block.size) + ") and 'n_block*n_grid' (" + std::to_string(N_B*M) + ") must match");
            if (buf_w_block.shape[0] != N_B)
                throw_with_line_number("Number of input w ("     + std::to_string(buf_w_block.shape[0]) + ") and 'n_block' (" + std::to_string(N_B) + ") must match");
        }
        else
            throw_with_line_number("input w have wrong dimension (" + std::to_string(buf_w_block.ndim) +"), note that spacial dimensions of fields should be flattened");
        

        try{
            double single_partition;
            py::array_t<double> phi = py::array_t<double>(N_B*M);
            py::buffer_info buf_phi = phi.request();
            
            compute_statistics((double*) buf_phi.ptr,
                    (double*) buf_q1_init.ptr, (double*) buf_q2_init.ptr,
                    (double*) buf_w_block.ptr, single_partition);
            return std::make_tuple(std::move(phi), single_partition);
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    };
    std::tuple<py::array_t<double>, py::array_t<double>> get_partition(int n1, int n2)
    {
        try{
            const int M = cb->get_n_grid();
            py::array_t<double> q1 = py::array_t<double>(M);
            py::array_t<double> q2 = py::array_t<double>(M);
            py::buffer_info buf_q1 = q1.request();
            py::buffer_info buf_q2 = q2.request();

            get_partition((double*) buf_q1.ptr, n1, (double*) buf_q2.ptr, n2);
            
            return std::make_tuple(std::move(q1), std::move(q2));
        }
        catch(std::exception& exc)
        {
            throw_with_line_number(exc.what());
        }
    };
};
#endif
