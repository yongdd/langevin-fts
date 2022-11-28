/*----------------------------------------------------------
* This class defines polyer chain parameters
*-----------------------------------------------------------*/

#ifndef POLYMER_CHAIN_H_
#define POLYMER_CHAIN_H_

#include <string>
#include <vector>
#include <map>

class PolymerChain
{
private:
    int n_block;         // total number of blocks
    int n_segment_total; // total number of segments, N_p
    double ds;           // contour step interval
    std::string model_name;         // "Continuous": continuous standard Gaussian model
                                    // "Discrete": discrete bead-spring model
    std::vector<std::string> types;         // sequence of block types, e.g., ["A","B",...]
    std::vector<int> n_segments;            // sequence of block segments number, e.g., [N_A, N_B, ...]
    std::vector<double> bond_length_sq;    // square of sequence of statistical_segment_length [(a_A/a_Ref)^2, (a_B/a_Ref)^2, ...]

    std::vector<int> block_start;   // index of starting segment of each block
                                    // [0,N_A,N_A+N_B,...,N] (length = n_block+1)

    //std::vector<int> block_lengths;     // sequence of block lengths, e.g., [0.3, 0.4, ...]
public:
    PolymerChain(std::vector<std::string> types, std::vector<double> block_lengths,
        std::map<std::string, double> dict_segment_lengths, double ds, std::string model_name);
    ~PolymerChain() {};

    int get_n_block();
    std::vector<int> get_n_segment();    // [N_A, N_B, ...]
    int get_n_segment(int block);
    int get_n_segment_total();
    double get_ds();
    std::vector<double> get_bond_length_sq();
    double get_bond_length_sq(int block);
    std::vector<std::string> get_type();
    std::string get_type(int block);
    std::string get_model_name();

    std::vector<int> get_block_start();
};
#endif
