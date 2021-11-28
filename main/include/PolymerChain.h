/*----------------------------------------------------------
* This class defines polyer chain parameters
*-----------------------------------------------------------*/

#ifndef POLYMER_CHAIN_H_
#define POLYMER_CHAIN_H_

#include <string>

class PolymerChain
{
private:
    int n_contour;  // number of contour steps, N
    int n_contour_a;  // number of contour steps for A fraction, N_A
    double f; // A fraction (1-f is the B fraction)
    double ds;  // discrete step sizes
    double chi_n; // chi N, interaction parameter between A and B Monomers

public:
    // std::string type; 
    // Gaussian: continous standard Gaussian model
    // Discrete: discrete bead-spring model
    
    PolymerChain(double f, int n_contour, double chi_n);
    ~PolymerChain() {};

    int get_n_contour();    // N
    int get_n_contour_a();  // N_A
    int get_n_contour_b();  // N_B
    double get_f();
    double get_ds();
    double get_chi_n();
    //void set_chi_n(double chi_n);
};
#endif
