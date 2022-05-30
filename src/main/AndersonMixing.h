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
    virtual void caculate_new_fields(
        double *w, double *w_out, double *w_deriv,
        double old_error_level, double error_level)=0;

    // Methods for pybind11
    void caculate_new_fields(py::array_t<double> w, py::array_t<double> w_out, py::array_t<double> w_deriv,
                             double old_error_level, double error_level)
    {
        try{
            py::buffer_info buf_w = w.request();
            py::buffer_info buf_w_out = w_out.request();
            py::buffer_info buf_w_deriv = w_deriv.request();

            if (buf_w.size != n_var)
                throw_with_line_number("Size of input w ("       + std::to_string(buf_w.size)       + ") and 'n_var' (" + std::to_string(n_var) + ") must match");
            if (buf_w_out.size != n_var)
                throw_with_line_number("Size of input w_out ("   + std::to_string(buf_w_out.size)   + ") and 'n_var' (" + std::to_string(n_var) + ") must match");
            if (buf_w_deriv.size != n_var)
                throw_with_line_number("Size of input w_deriv (" + std::to_string(buf_w_deriv.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match");

            caculate_new_fields((double *) buf_w.ptr, (double *) buf_w_out.ptr, (double *) buf_w_deriv.ptr, old_error_level, error_level);
        }
        catch(std::exception& exc)
        {
            std::cerr << exc.what() << std::endl;
            throw_with_line_number("");
        }
    };

};
#endif
