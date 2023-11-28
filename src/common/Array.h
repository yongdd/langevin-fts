/*-------------------------------------------------------------
* This is an abstract Array class.
  It is made to perform array-wise arithmetic operations.
*--------------------------------------------------------------*/
#ifndef ARRAY_H_
#define ARRAY_H_

#include <iostream>
#include <vector>

#include "Exception.h"

class Array
{
protected:
    std::string device;
    unsigned int size = 0;
    int device_id;
public:
    virtual ~Array() {};

    // Overloading for array operation
    virtual void add(const Array& src_1, const Array& src_2)=0;
    virtual void subtract(const Array& src_1, const Array& src_2)=0;
    virtual void multiply(const Array& src_1, const Array& src_2)=0;
    virtual void divide(const Array& src_1, const Array& src_3)=0;

    // Arithmetic operations with a float number
    virtual void linear_scaling(const Array& src, const double a, const double b)=0;

    // Copy assignment 
    virtual void operator=(const Array&)=0;
    virtual void set_data(double *, unsigned int)=0;

    // // return concatenation of this and arr_b
    // Virtual void concatenate(const Array&)=0;

    // Return array as vector
    virtual std::vector<double> to_vector() const=0;

    // Return data pointer
    virtual double* get_ptr() const=0;

    // Return size of data
    virtual unsigned int get_size() const=0;
    
    // Access element of array
    virtual double operator[](unsigned int) const=0;
};

// // Arithmetic operations with a float number
// Array& operator+(const double a, const Array& array);
// Array& operator-(const double a, const Array& array);
// Array& operator*(const double a, const Array& array);
// Array& operator/(const double a, const Array& array);

#endif