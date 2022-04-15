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
    double epsilon; // epsilon = a_A/a_B, conformational asymmetry
                    // a = sqrt(f*a_A^2 + (1-f)*a_B^2)
    std::string model_name;   // "Gaussian": continous standard Gaussian model
                              // "Discrete": discrete bead-spring model
public:

    PolymerChain(double f, int n_contour, double chi_n, std::string model_name, double epsilon);
    ~PolymerChain() {};

    int get_n_contour();    // N
    int get_n_contour_a();  // N_A
    int get_n_contour_b();  // N_B
    double get_f();
    double get_ds();
    double get_chi_n();
    double get_epsilon();
    std::string get_model_name();

};
#endif
