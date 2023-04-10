/*-------------------------------------------------------------
* This is an abstract Array class.
  It is made to perform array-wise arithmetic operations.
*--------------------------------------------------------------*/
#ifndef ARRAY_H_
#define ARRAY_H_

#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Exception.h"

namespace py = pybind11;

class Array
{
protected:
    std::string device;
    unsigned int size = 0;
    int device_id;
public:
    virtual ~Array() {};

    // overloading for array operation
    virtual void add(const Array& src_1, const Array& src_2)=0;
    virtual void subtract(const Array& src_1, const Array& src_2)=0;
    virtual void multiply(const Array& src_1, const Array& src_2)=0;
    virtual void divide(const Array& src_1, const Array& src_3)=0;

    // arithmetic operations with a float number
    virtual void linear_scaling(const Array& src, const double a, const double b)=0;

    // copy assignment 
    virtual void operator=(const Array&)=0;
    virtual void set_data(double *, unsigned int)=0;

    // // return concatenation of this and arr_b
    // virtual void concatenate(const Array&)=0;

    // return array as vector
    virtual std::vector<double> to_vector() const=0;

    // return data pointer
    virtual double* get_ptr() const=0;

    // return size of data
    virtual unsigned int get_size() const=0;
    
    // access element of array
    virtual double operator[](unsigned int) const=0;

    // Methods for pybind11
    void set_data_pybind11(py::array_t<const double> data)
    {
        try
        {
            py::buffer_info buf = data.request();
            if (buf.size != this->size) {
                throw_with_line_number("Size of input (" + std::to_string(buf.size) + ") and 'n_grid' (" + std::to_string(this->size) + ") must match");
            }
            set_data((double*) buf.ptr, this->size);
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    }
    long int get_ptr_pybind11()
    {
        try
        {
            double* data_ptr = get_ptr();
            return reinterpret_cast<std::uintptr_t>(data_ptr);
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    }
};

// // arithmetic operations with a float number
// Array& operator+(const double a, const Array& array);
// Array& operator-(const double a, const Array& array);
// Array& operator*(const double a, const Array& array);
// Array& operator/(const double a, const Array& array);

#endif