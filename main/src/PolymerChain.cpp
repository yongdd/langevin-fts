#include <iostream>
#include <cmath>
#include "PolymerChain.h"

//----------------- Constructor ----------------------------
PolymerChain::PolymerChain(double f, int n_contour, double chi_n)
{
    this->f = f;
    this->n_contour = n_contour;
    this->chi_n = chi_n;

    // grid number for A fraction
    this->n_contour_a = std::lround(n_contour*f);
    if( std::abs(this->n_contour_a-n_contour*f) > 1.e-6)
    {
        std::cerr<< "N*f is not an integer"<< std::endl;
        exit(-1);
    }
    // grid sizes contour direction
    this->ds = 1.0/n_contour;
}
int PolymerChain::get_n_contour()
{
    return n_contour;
}
int PolymerChain::get_n_contour_a()
{
    return n_contour_a;
}
int PolymerChain::get_n_contour_b()
{
    return n_contour - n_contour_a;
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
