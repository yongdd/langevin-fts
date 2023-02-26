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
    int max_n_segment;                                    // the maximum segment number
    std::string monomer_type;                             // monomer_type
    std::vector<std::tuple<std::string, int, int>> deps;  // tuple <key, n_segment, n_repeated>
    int height;                                           // height of branch (height of tree data Structure)
};
struct UniqueBlock{
    std::string monomer_type;  // monomer_type

    // When the 'use_superposition' is on, original block can be sliced to smaller block pieces.
    // For example, suppose one block is composed of 5'A' monomers. -> -A-A-A-A-A-
    // And this block is sliced into 3'A' monomers and 2'A' monomers -> -A-A-A-,  -A-A-
    // For the first slice, n_segment_allocated, n_segment_offset, and n_segment_original are 3, 0, and 5, respectively.
    // For the second slice, n_segment_allocated, n_segment_offset, and n_segment_original are 2, 3, and 5, respectively.
    // If the 'use_superposition' is off, original block is not sliced to smaller block pieces.
    // In this case, n_segment_allocated, n_segment_offset, and n_segment_original are 5, 0, and 5, respectively.

    int n_segment_allocated;
    int n_segment_offset;      
    int n_segment_original;
    std::vector<std::tuple<int ,int>> v_u; // node pair <polymer id, v, u>
};

/* This stucture defines comparison function for branched key */
struct CompareBranchKey
{
    bool operator()(const std::string& str1, const std::string& str2);
};

class Mixture
{
private:
    std::string model_name; // "continuous": continuous standard Gaussian model
                            // "discrete": discrete bead-spring model
                            
    double ds;              // contour step interval
    bool use_superposition; // compute multiple partial partition functions using property of linearity of the diffusion equation.

    // dictionary{key:monomer_type, value:relative statistical_segment_length. (a_A/a_Ref)^2 or (a_B/a_Ref)^2, ...}
    std::map<std::string, double> bond_lengths;

    // distinct_polymers
    std::vector<PolymerChain> distinct_polymers;

    // set{key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u))}
    std::map<std::tuple<int, std::string, std::string>, UniqueBlock> unique_blocks;

    // dictionary{key:non-duplicated Unique sub_branches, value: UniqueEdge}
    std::map<std::string, UniqueEdge, CompareBranchKey> unique_branches; 

    // get text code of branch
    std::pair<std::string, int> get_text_code_of_branch(
        std::vector<PolymerChainBlock> blocks,
        std::map<int, std::vector<int>> adjacent_nodes,
        std::map<std::pair<int, int>, int> edge_to_array,
        std::map<int, std::string> chain_end_to_q_init,
        int in_node, int out_node);

    // add new key. if it already exists and 'new_n_segment' is larger than 'max_n_segment', update it.
    void add_unique_branch(std::map<std::string, UniqueEdge, CompareBranchKey>& unique_branches, std::string new_key, int new_n_segment);

    // superpose branches
    std::map<std::string, UniqueBlock> superpose_branches_common    (std::map<std::string, UniqueBlock> remaining_keys, int minimum_n_segment);
    std::map<std::string, UniqueBlock> superpose_branches_of_continuous_chain(std::map<std::string, UniqueBlock> u_map);
    std::map<std::string, UniqueBlock> superpose_branches_of_discrete_chain  (std::map<std::string, UniqueBlock> u_map);

public:

    Mixture(std::string model_name, double ds, std::map<std::string, double> bond_lengths, bool use_superposition);
    ~Mixture() {};

    std::string get_model_name() const;
    double get_ds() const;
    const std::map<std::string, double>& get_bond_lengths() const;
    bool is_using_superposition() const;

    // add new polymer

    // Mark some chain ends to set initial conditions of partial partition functions when pseudo.compute_statistics() is invoked.
    // For instance, if chain_end_to_q_init[a] is set to b, 
    // q_init[b] will be used as an initial condition of chain end 'a' in pseudo.compute_statistics().
        void add_polymer(
        double volume_fraction,
        std::vector<std::string> block_monomer_types,
        std::vector<double> contour_lengths,
        std::vector<int> v, std::vector<int> u,
        std::map<int, std::string> chain_end_to_q_init);

    // add new polymer (All chain ends are free ends, e.g, q(r,0) = 1)
    void add_polymer(
        double volume_fraction,
        std::vector<std::string> block_monomer_types,
        std::vector<double> contour_lengths,
        std::vector<int> v, std::vector<int> u)
    {
        add_polymer(volume_fraction, block_monomer_types, contour_lengths, v, u, {});
    }

    // get polymers
    int get_n_polymers() const;
    PolymerChain& get_polymer(const int p);

    // get information of unique branches and blocks
    int get_unique_n_branches() const;
    std::map<std::string, UniqueEdge, CompareBranchKey>& get_unique_branches(); 
    UniqueEdge& get_unique_branch(std::string key);
    std::map<std::tuple<int, std::string, std::string>, UniqueBlock>& get_unique_blocks(); 
    UniqueBlock& get_unique_block(std::tuple<int, std::string, std::string> key);

    // get information from key
    static std::vector<std::tuple<std::string, int, int>> get_deps_from_key(std::string key);
    static std::string remove_monomer_type_from_key(std::string key);
    static std::string get_monomer_type_from_key(std::string key);
    static std::string get_q_input_idx_from_key(std::string key);
    static int get_height_from_key(std::string key);

    // display
    void display_unique_branches() const;
    void display_unique_blocks() const;
    void display_all_unique_branch_deps() const;
};
#endif
