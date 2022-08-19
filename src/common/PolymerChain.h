/*----------------------------------------------------------
* This class defines polyer chain parameters
*-----------------------------------------------------------*/

#ifndef POLYMER_CHAIN_H_
#define POLYMER_CHAIN_H_

#include <string>
#include <vector>

class PolymerChain
{
private:
    int n_block;
    int n_segment_total; // number of segements, N
    std::vector<int> n_segment;  // number of segements, [N_A, N_B, ...]
    std::vector<double> bond_length;  // bond length of each blocks [a_A, a_B, ...]
    double ds;  // discrete step sizes
    //double epsilon; // epsilon = a_A/a_B, conformational asymmetry
                    // a = sqrt(f*a_A^2 + (1-f)*a_B^2)
    std::string model_name;   // "Continuous": continous standard Gaussian model
                              // "Discrete": discrete bead-spring model
    std::vector<int> block_start; // index of starting segment of each block
                        // [0,N_A,N_A+N_B,...,N] (length = n_block+1)
public:

    PolymerChain(std::vector<int> n_segment, std::vector<double> bond_length, std::string model_name);
    ~PolymerChain() {};

    int get_n_block();
    std::vector<int> get_n_segment();    // [N_A, N_B, ...]
    int get_n_segment(int block);
    int get_n_segment_total();
    double get_ds();
    std::vector<double> get_bond_length();
    double get_bond_length(int block);
    std::string get_model_name();

    std::vector<int> get_block_start();
};
#endif
