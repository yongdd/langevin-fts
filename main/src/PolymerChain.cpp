#include <iostream>
#include <cmath>
#include "PolymerChain.h"

//----------------- Constructor ----------------------------
PolymerChain::PolymerChain(double f, int NN, double chi_n)
{
    this->f = f;
    this->NN = NN;
    this->chi_n = chi_n;

    // grid number for A fraction
    this->NN_A = std::lround(NN*f);
    if( std::abs(this->NN_A-NN*f) > 1.e-6)
    {
        std::cerr<< "NN*f is not an integer"<< std::endl;
        exit(-1);
    }
    // grid sizes contour direction
    this->ds = 1.0/NN;
}
int PolymerChain::get_NN()
{
    return NN;
}
int PolymerChain::get_NN_A()
{
    return NN_A;
}
int PolymerChain::get_NN_B()
{
    return NN - NN_A;
}
double PolymerChain::get_f()
{
    return f;
}
double PolymerChain::get_ds()
{
    return ds;
}
double PolymerChain::get_chi_n()
{
    return chi_n;
}
////----------------- set_chi_n ----------------------------
//void PolymerChain::set_chin(double chi_n)
//{
//this->chi_n = chi_n;
//}
