/*----------------------------------------------------------
* This class defines branched polymer chain parameters
*-----------------------------------------------------------*/

#ifndef BRANCHED_POLYMER_CHAIN_H_
#define BRANCHED_POLYMER_CHAIN_H_

#include <string>
#include <vector>
#include <map>

struct polymer_chain_block{
    std::string species;    // species of block, e.g., "A" or "B", ...
    int n_segment;          // the number of segment, e.g. 20, or 30 ...
    double bond_length_sq;  // square of relative statistical_segment_length (a_A/a_Ref)^2 or (a_B/a_Ref)^2, ...
    double contour_length;  // relative length of each block, e.g., 1.0, 0.4, 0.3, ...
    int v;                  // starting vertex (or node)
    int u;                  // ending vertex (or node)
    std::string dep_v;      // forward direction prerequisite partial partition functions as a text
    std::string dep_u;      // backward direction prerequisite partial partition functions as a text
};

class BranchedPolymerChain
{
private:
    std::string model_name; // "Continuous": continuous standard Gaussian model
                            // "Discrete": discrete bead-spring model
    double ds;              // contour step interval

    // Lists of blocks,
    // <node, node2, species, block_length>
    // e.g., <0, 1, "A", 0.5>
    std::vector<polymer_chain_block> blocks;  // information of blocks, which contains 'species', 'n_segments', '
                                              // bond_length_sq', 'contour_length', 'vertices', and 'dependencies'.

    std::map<int, std::vector<int>> adjacent_nodes;     // adjacent nodes
    std::map<std::pair<int, int>, int> edge_to_array;   // array index for each edge

    // dictionary{key:non-duplicated optimal sub_branches, value:maximum segment number}
    std::map<std::string, int, std::greater<std::string>> opt_max_segments; 

    // get sub-branch information as ordered texts
    std::pair<std::string, int> get_text_of_ordered_branches(int in_node, int out_node);
public:

    BranchedPolymerChain(std::string model_name, double ds, std::map<std::string, double> dict_segment_lengths,
        std::vector<std::string> block_species, std::vector<double> contour_lengths, std::vector<int> v, std::vector<int> u);
    ~BranchedPolymerChain() {};

    std::string get_model_name();
    double get_ds();

    int get_n_block();
    std::string get_block_type(int idx);
    int get_n_segment(int idx);
    double get_bond_length_sq(int idx);

    //std::vector<std::string> get_block_type();
    //std::vector<int> get_n_segment();    // [N_A, N_B, ...]
    //std::vector<double> get_bond_length_sq();
    //std::vector<std::pair<std::string, int>> get_opt_sub_deps(std::string key);

    // get information of optimal sub_graph
    int get_opt_n_block();
    int get_opt_n_branches();
    std::vector<std::pair<std::string, int>> key_to_deps(std::string key);
    std::string key_to_species(std::string key);
    std::map<std::string, int, std::greater<std::string>> get_opt_max_segments(); 
    int get_opt_max_segment(std::string key);
};
#endif