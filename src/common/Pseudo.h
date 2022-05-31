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

#include "SimulationBox.h"
#include "PolymerChain.h"
#include "Exception.h"

namespace py = pybind11;

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

    // Methods for pybind11
    std::tuple<py::array_t<double>, py::array_t<double>, double>
    find_phi(py::array_t<double> q1_init, py::array_t<double> q2_init, py::array_t<double> w_a, py::array_t<double> w_b)
    {
        const int M = sb->get_n_grid();
        py::buffer_info buf_q1_init = q1_init.request();
        py::buffer_info buf_q2_init = q2_init.request();
        py::buffer_info buf_w_a = w_a.request();
        py::buffer_info buf_w_b = w_b.request();

        if (buf_q1_init.size != M)
            throw_with_line_number("Size of input q1_init (" + std::to_string(buf_q1_init.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
        if (buf_q2_init.size != M)
            throw_with_line_number("Size of input q2_init (" + std::to_string(buf_q2_init.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
        if (buf_w_a.size != M)
            throw_with_line_number("Size of input w_a ("     + std::to_string(buf_w_a.size)     + ") and 'n_grid' (" + std::to_string(M) + ") must match");
        if (buf_w_b.size != M)
            throw_with_line_number("Size of input w_b ("     + std::to_string(buf_w_b.size)     + ") and 'n_grid' (" + std::to_string(M) + ") must match");

        try{
            double single_partition;
            py::array_t<double> phi_a = py::array_t<double>(M);
            py::array_t<double> phi_b = py::array_t<double>(M);
            py::buffer_info buf_phi_a = phi_a.request();
            py::buffer_info buf_phi_b = phi_b.request();

            find_phi((double*) buf_phi_a.ptr,   (double*) buf_phi_b.ptr,
                    (double*) buf_q1_init.ptr, (double*) buf_q2_init.ptr,
                    (double*) buf_w_a.ptr,     (double*) buf_w_b.ptr, single_partition);
            
            return std::make_tuple(std::move(phi_a), std::move(phi_b), single_partition);
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    };
    std::tuple<py::array_t<double>, py::array_t<double>> get_partition(int n1, int n2)
    {
        try{
            const int M = sb->get_n_grid();
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
