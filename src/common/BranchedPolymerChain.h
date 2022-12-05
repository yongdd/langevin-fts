/*----------------------------------------------------------
* This class defines branched polymer chain parameters
*-----------------------------------------------------------*/

#ifndef BRANCHED_POLYMER_CHAIN_H_
#define BRANCHED_POLYMER_CHAIN_H_

#include <string>
#include <vector>
#include <map>

struct polymer_chain_edge_data{
    std::string species;        // block species
    std::string dependency;     // prerequisite partial partition functions as a text
    int n_segment;              // the number of segments
};

struct polymer_chain_optimal_sub_branches_data{
    std::vector<std::pair<std::string, int>> dependency;  // prerequisite partial partition functions as texts
    int max_segment;                                        // the maximum segment length for given branches
};

class BranchedPolymerChain
{
private:
    std::string model_name; // "Continuous": continuous standard Gaussian model
                            // "Discrete": discrete bead-spring model
    double ds;              // contour step interval

    // Lists of block edges,
    // <node, node2, species, block_length>
    // e.g., <0, 1, "A", 0.5>
    //std::vector<std::map<int, int, std::string, double>> blocks;
    std::vector<std::string> block_types;         // sequence of block types, e.g., ["A","B",...]
    std::vector<int> n_segments;            // sequence of block segments number, e.g., [N_A, N_B, ...]
    std::vector<double> bond_length_sq;     // square of sequence of statistical_segment_length [(a_A/a_Ref)^2, (a_B/a_Ref)^2, ...]

    //std::vector<int> v_edges, u_edges;             // v and u of edge
    std::map<int, std::vector<int>> adjacent_nodes;                     // 
    std::map<std::pair<int, int>, polymer_chain_edge_data> edges;   // species, segments and dependency for each edge

    // non-duplicated optimal sub_branches
    std::map<std::string, polymer_chain_optimal_sub_branches_data, std::greater<std::string>> optimal_sub_branches; 

    // get sub-branch information as ordered texts
    std::pair<std::string, int> get_text_of_ordered_branches(int in_node, int out_node);
public:

    BranchedPolymerChain(std::string model_name, double ds, std::map<std::string, double> dict_segment_lengths,
        std::vector<std::string> block_types, std::vector<double> block_lengths, std::vector<int> v, std::vector<int> u);
    ~BranchedPolymerChain() {};

    std::string get_model_name();
    double get_ds();

    std::vector<double> get_bond_length_sq();
    double get_bond_length_sq(int block);
    int get_n_block();
    std::vector<int> get_n_segment();    // [N_A, N_B, ...]
    int get_n_segment(int block);

    std::vector<std::string> get_block_type();
    std::string get_block_type(int block_number);

    // get information of optimal sub_graph
    std::map<std::string, polymer_chain_optimal_sub_branches_data, std::greater<std::string>> get_optimal_sub_branches(); 
    int get_n_sub_graph();
    std::vector<std::pair<std::string, int>> get_dependency(std::string key);
    int get_max_segment(std::string key);
};
#endif
