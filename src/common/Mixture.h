/*----------------------------------------------------------
* This class defines branched polymer chain parameters
*-----------------------------------------------------------*/

#ifndef MIXTURE_H_
#define MIXTURE_H_

#include <string>
#include <vector>
#include <map>

#include "PolymerChain.h"

struct ReducedEdge{
    int max_n_segment;                              // the maximum segment number
    std::string species;                            // species
    std::vector<std::pair<std::string, int>> deps;  // dependency pairs
};

struct ReducedBlock{
    std::string species;  // species
};

class Mixture
{
private:
    std::string model_name; // "Continuous": continuous standard Gaussian model
                            // "Discrete": discrete bead-spring model
    double ds;              // contour step interval

    // dictionary{key:species, value:relative statistical_segment_length. (a_A/a_Ref)^2 or (a_B/a_Ref)^2, ...}
    std::map<std::string, double> bond_lengths;

    // distinct_polymers
    std::vector<PolymerChain *> distinct_polymers;

    // dictionary{key:non-duplicated reduced sub_branches, value: ReducedEdge}
    std::map<std::string, ReducedEdge, std::greater<std::string>> reduced_branches; 

    // set{key: (dep_v, dep_u) (assert(dep_v <= dep_u))}
    std::map<std::tuple<std::string, std::string, int>, ReducedBlock> reduced_blocks; 

    // get sub-branch information as ordered texts
    std::pair<std::string, int> get_text_of_ordered_branches(
        std::vector<PolymerChainBlock> blocks,
        std::map<int, std::vector<int>> adjacent_nodes,
        std::map<std::pair<int, int>, int> edge_to_array,
        int in_node, int out_node);
public:

    Mixture(std::string model_name, double ds, std::map<std::string, double> bond_lengths);
    ~Mixture();

    std::string get_model_name();
    double get_ds();
    std::map<std::string, double>& get_bond_lengths();

    // distinct_polymers
    void add_polymer_chain(
        double volume_fraction,
        std::vector<std::string> block_species,
        std::vector<double> contour_lengths,
        std::vector<int> v, std::vector<int> u,
        std::map<int, int> v_to_grafting_index);
    int get_n_distinct_polymers();
    PolymerChain* get_polymer_chain(int p);

    // get information of reduced sub graphs
    int get_reduced_n_branches();
    std::vector<std::pair<std::string, int>> key_to_deps(std::string key);
    std::string key_to_species(std::string key);
    std::map<std::string, ReducedEdge, std::greater<std::string>>& get_reduced_branches(); 
    ReducedEdge get_reduced_branch(std::string key);
    std::map<std::tuple<std::string, std::string, int>, ReducedBlock>& get_reduced_blocks(); 
    ReducedBlock get_reduced_block(std::tuple<std::string, std::string, int> key);
};
#endif
