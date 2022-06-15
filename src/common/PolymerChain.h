/*----------------------------------------------------------
* This class defines polyer chain parameters
*-----------------------------------------------------------*/

#ifndef POLYMER_CHAIN_H_
#define POLYMER_CHAIN_H_

#include <string>

class PolymerChain
{
private:
    int n_segment;  // number of segements, N
    int n_segment_a;  // number of A segements, N_A
    double f; // A fraction (1-f is the B fraction)
    double ds;  // discrete step sizes
    double chi_n; // chi N, interaction parameter between A and B Monomers
    double epsilon; // epsilon = a_A/a_B, conformational asymmetry
                    // a = sqrt(f*a_A^2 + (1-f)*a_B^2)
    std::string model_name;   // "Continuous": continous standard Gaussian model
                              // "Discrete": discrete bead-spring model
public:

    PolymerChain(double f, int n_segment, double chi_n, std::string model_name, double epsilon);
    ~PolymerChain() {};

    int get_n_segment();    // N
    int get_n_segment_a();  // N_A
    int get_n_segment_b();  // N_B
    double get_f();
    double get_ds();
    double get_chi_n();
    double get_epsilon();
    std::string get_model_name();

    void set_chi_n(double chi_n);
};
#endif
