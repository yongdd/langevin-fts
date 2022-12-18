/*-------------------------------------------------------------
* This is an abstract AndersonMixing class
*------------------------------------------------------------*/

#ifndef ANDERSON_MIXING_H_
#define ANDERSON_MIXING_H_

#include <cassert>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Exception.h"

namespace py = pybind11;

class AndersonMixing
{
protected:
    int n_var, max_hist, n_anderson;
    double start_error, mix_min, mix, mix_init;

    void find_an(double **u, double *v, double *a, int n);
public:
    AndersonMixing(int n_var, int max_hist, double start_error, double mix_min, double mix_init);
    virtual ~AndersonMixing(){};

    virtual void reset_count(){};
    virtual void calculate_new_fields(
        double *w_new, double *w_current, double *w_deriv,
        double old_error_level, double error_level)=0;

    // Methods for pybind11
    py::array_t<double> calculate_new_fields(py::array_t<double> w_current, py::array_t<double> w_deriv,
                             double old_error_level, double error_level)
    {
        try{

            py::array_t<double> w_new = py::array_t<double>(n_var);

            py::buffer_info buf_w_new = w_new.request();
            py::buffer_info buf_w_current = w_current.request();
            py::buffer_info buf_w_deriv = w_deriv.request();

            if (buf_w_new.size != n_var)
                throw_with_line_number("Size of input w_new (" + std::to_string(buf_w_new.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match");
            if (buf_w_current.size != n_var)
                throw_with_line_number("Size of input w_current (" + std::to_string(buf_w_current.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match");
            if (buf_w_deriv.size != n_var)
                throw_with_line_number("Size of input w_deriv (" + std::to_string(buf_w_deriv.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match");

            calculate_new_fields((double *) buf_w_new.ptr, (double *) buf_w_current.ptr, (double *) buf_w_deriv.ptr, old_error_level, error_level);
            return std::move(w_new);
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    };

};
#endif
