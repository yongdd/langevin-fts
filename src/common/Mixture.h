/*----------------------------------------------------------
* This class defines polymer mixture parameters
*-----------------------------------------------------------*/

#ifndef MIXTURE_H_
#define MIXTURE_H_

#include <string>
#include <vector>
#include <map>

#include "PolymerChain.h"

struct UniqueEdge{
    int max_n_segment;                              // the maximum segment number
    std::string monomer_type;                       // monomer_type
    std::vector<std::pair<std::string, int>> deps;  // dependency pairs
    int height;                                     // height of branch (height of tree data Structure)
};
struct UniqueBlock{
    std::string monomer_type;  // monomer_type
};

class Mixture
{
private:
    std::string model_name; // "Continuous": continuous standard Gaussian model
                            // "Discrete": discrete bead-spring model
    double ds;              // contour step interval

    // dictionary{key:monomer_type, value:relative statistical_segment_length. (a_A/a_Ref)^2 or (a_B/a_Ref)^2, ...}
    std::map<std::string, double> bond_lengths;

    // distinct_polymers
    std::vector<PolymerChain> distinct_polymers;

    // dictionary{key:non-duplicated Unique sub_branches, value: UniqueEdge}
    std::map<std::string, UniqueEdge, std::greater<std::string>> unique_branches; 

    // set{key: (dep_v, dep_u) (assert(dep_v <= dep_u))}
    std::map<std::tuple<std::string, std::string, int>, UniqueBlock> unique_blocks; 

    // get sub-branch information as ordered texts
    std::pair<std::string, int> get_text_of_ordered_branches(
        std::vector<PolymerChainBlock> blocks,
        std::map<int, std::vector<int>> adjacent_nodes,
        std::map<std::pair<int, int>, int> edge_to_array,
        int in_node, int out_node);
public:

    Mixture(std::string model_name, double ds, std::map<std::string, double> bond_lengths);
    ~Mixture() {};

    std::string get_model_name() const;
    double get_ds() const;
    const std::map<std::string, double>& get_bond_lengths() const;

    // distinct_polymers
    void add_polymer(
        double volume_fraction,
        std::vector<std::string> block_monomer_types,
        std::vector<double> contour_lengths,
        std::vector<int> v, std::vector<int> u,
        std::map<int, int> v_to_grafting_index);
    int get_n_polymers() const;
    PolymerChain& get_polymer(const int p);

    // get information of Unique sub graphs
    int get_unique_n_branches() const;
    static std::vector<std::pair<std::string, int>> key_to_deps(std::string key);
    static std::string key_to_species(std::string key);
    std::map<std::string, UniqueEdge, std::greater<std::string>>& get_unique_branches(); 
    UniqueEdge& get_unique_branch(std::string key);
    std::map<std::tuple<std::string, std::string, int>, UniqueBlock>& get_unique_blocks(); 
    UniqueBlock& get_unique_block(std::tuple<std::string, std::string, int> key);

    void display_unique_branches() const;
    void display_unique_blocks() const;

    // Methods for pybind11
    void add_polymer(double volume_fraction,
                    std::vector<std::string> block_monomer_types,
                    std::vector<double> contour_lengths,
                    std::vector<int> v, std::vector<int> u)
    {
        add_polymer(volume_fraction, block_monomer_types, contour_lengths, v, u, {});
    }
};
#endif
