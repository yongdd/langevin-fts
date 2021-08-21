/*----------------------------------------------------------
* This class defines polyer chain parameters
*-----------------------------------------------------------*/

#ifndef POLYMER_CHAIN_H_
#define POLYMER_CHAIN_H_

#include <string>

class PolymerChain
{
private:
    int NN;  // number of contour steps
    int NN_A;  // number of contour steps for A fraction
    double f; // A fraction (1-f is the B fraction)
    double ds;  // discrete step sizes
    double chi_n; // chi N, interaction parameter between A and B Monomers

public:
    //const std::string type = "Gaussian"; 
    // Gaussian: continous standard Gaussian model
    // Discrete: discrete bead-spring model
    
    PolymerChain(double f, int NN, double chi_n);
    ~PolymerChain() {};

    int get_NN();
    int get_NN_A();
    int get_NN_B();
    double get_f();
    double get_ds();
    double get_chi_n();
    //void set_chi_n(double chi_n);
};
#endif
