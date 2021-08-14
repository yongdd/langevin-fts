/*----------------------------------------------------------
* This class defines polyer chain parameters
*-----------------------------------------------------------*/

#ifndef POLYMER_CHAIN_H_
#define POLYMER_CHAIN_H_

#include <iostream>
#include <cmath>

class PolymerChain
{
private:
public:
    int NN;  // number of contour steps
    int NNf;  // number of contour steps for A fraction
    double f; // A fraction (1-f is the B fraction)
    double ds;  // discrete step sizes
    double chi_n; // chi N, interaction parameter between A and B Monomers

    PolymerChain(double f, int NN, double chi_n);
    ~PolymerChain() {};

    void set_chin(double chi_n);
};
#endif
