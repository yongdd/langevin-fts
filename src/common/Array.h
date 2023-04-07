/*-------------------------------------------------------------
* This is an abstract Array class.
  It is made to perform array-wise arithmetic operations.
*--------------------------------------------------------------*/
#ifndef ARRAY_H_
#define ARRAY_H_

#include <vector>

#include "Exception.h"

class Array
{
protected:
    std::string device;
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
};

// // arithmetic operations with a float number
// Array& operator+(const double a, const Array& array);
// Array& operator-(const double a, const Array& array);
// Array& operator*(const double a, const Array& array);
// Array& operator/(const double a, const Array& array);

#endif