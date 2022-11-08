/*-------------------------------------------------------------
* This is an abstract ComputationBox class.
* This class defines Computation Grids and Lengths parameters, and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#ifndef COMPUTATION_BOX_H_
#define COMPUTATION_BOX_H_

#include <array>
#include <vector>
#include <cassert>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Exception.h"

namespace py = pybind11;

class ComputationBox
{
protected:
    int dim;        // the dimension of Computation Grids and Lengths
    std::array<int,3> nx;  // the number of grid in each direction
    std::array<double,3> lx;  // length of the block copolymer in each direction (in units of aN^1/2)
    std::array<double,3> dx;  // grid interval in each direction
    int n_grid;  // the number of grid
    double *dv; // dV, simple integral weight,
    double volume; // volume of the system.

public:
    ComputationBox(std::vector<int> nx, std::vector<double> lx);
    virtual ~ComputationBox();

    int get_dim();
    std::array<int,3> get_nx();
    int get_nx(int i);
    std::array<double,3> get_lx();
    double get_lx(int i);
    std::array<double,3> get_dx();
    double get_dx(int i);
    double get_dv(int i);
    int get_n_grid();
    double get_volume();

    virtual void set_lx(std::vector<double> new_lx);

    double integral(double *g);
    double inner_product(double *g, double *h);
    double multi_inner_product(int n_comp, double *g, double *h);
    void zero_mean(double *g);

    // Methods for pybind11
    double integral(py::array_t<double> g) {
        py::buffer_info buf = g.request();
        if (buf.size != n_grid) {
            throw_with_line_number("Size of input (" + std::to_string(buf.size) + ") and 'n_grid' (" + std::to_string(n_grid) + ") must match");
        }
        return integral((double*) buf.ptr);
    };
    double inner_product(py::array_t<double> g, py::array_t<double> h) {
        py::buffer_info buf1 = g.request();
        py::buffer_info buf2 = h.request();
        if (buf1.size != n_grid) 
            throw_with_line_number("Size of input g (" + std::to_string(buf1.size) + ") and 'n_grid' (" + std::to_string(n_grid) + ") must match");
        if (buf2.size != n_grid)
            throw_with_line_number("Size of input h (" + std::to_string(buf2.size) + ") and 'n_grid' (" + std::to_string(n_grid) + ") must match");
        return inner_product((double*) buf1.ptr, (double*) buf2.ptr);
    };
    double multi_inner_product(int n_comp, py::array_t<double> g, py::array_t<double> h) {
        py::buffer_info buf1 = g.request();
        py::buffer_info buf2 = h.request();
        if (buf1.size != n_comp*n_grid) 
            throw_with_line_number("Size of input g (" + std::to_string(buf1.size) + ") and 'n_comp x n_grid' (" + std::to_string(n_comp*n_grid) + ") must match");
        if (buf2.size != n_comp*n_grid)
            throw_with_line_number("Size of input h (" + std::to_string(buf2.size) + ") and 'n_comp x n_grid' (" + std::to_string(n_comp*n_grid) + ") must match");
        return multi_inner_product(n_comp, (double*) buf1.ptr, (double*) buf2.ptr);
    };
    void zero_mean(py::array_t<double> g) {
        py::buffer_info buf = g.request();
        if (buf.size != n_grid) {
            throw_with_line_number("Size of input (" + std::to_string(buf.size) + ") and 'n_grid' (" + std::to_string(n_grid) + ") must match");
        }
        zero_mean((double*) buf.ptr);
    };
};
#endif
