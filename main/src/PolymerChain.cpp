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
    this->NNf = std::lround(NN*f);
    if( std::abs(this->NNf-NN*f) > 1.e-6)
    {
        std::cerr<< "NN*f is not an integer"<< std::endl;
        exit(-1);
    }
    // grid sizes contour direction
    this->ds = 1.0/NN;
}
//----------------- set_chi_n ----------------------------
void PolymerChain::set_chin(double chi_n)
{
    this->chi_n = chi_n;
}
