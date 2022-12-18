/*----------------------------------------------------------
* This class defines single polymer chain parameters
*-----------------------------------------------------------*/

#ifndef POLYMER_CHAIN_H_
#define POLYMER_CHAIN_H_

#include <string>
#include <vector>
#include <map>

struct PolymerChainBlock{
    std::string species;    // species of block, e.g., "A" or "B", ...
    int n_segment;          // the number of segment, e.g. 20, or 30 ...
    double contour_length;  // relative length of each block, e.g., 1.0, 0.4, 0.3, ...
    int v;                  // starting vertex (or node)
    int u;                  // ending vertex (or node)
};

class PolymerChain
{
private:
    double alpha;            // sum of contour lengths
    double volume_fraction;  // volume_fraction

    std::vector<PolymerChainBlock> blocks;  // information of blocks, which contains 'species', 'n_segments', '
                                            // bond_length_sq', 'contour_length', and 'vertices'.

    std::map<int, std::vector<int>> adjacent_nodes;             // adjacent nodes
    std::map<std::pair<int, int>, int>         edge_to_array;   // array index for each edge
    std::map<std::pair<int, int>, std::string> edge_to_deps;    // prerequisite partial partition functions as a text

public:
    PolymerChain(
        double ds, std::map<std::string, double> bond_lengths, 
        double volume_fraction, 
        std::vector<std::string> block_species,
        std::vector<double> contour_lengths,
        std::vector<int> v, std::vector<int> u,
        std::map<int, int> v_to_grafting_index={});
    ~PolymerChain() {};

    double get_alpha() const;
    double get_volume_fraction() const;

    int get_n_blocks() const;
    std::vector<PolymerChainBlock>& get_blocks();
    struct PolymerChainBlock& get_block(const int v, const int u);

    int get_n_segment_total() const;
    int get_n_segment(const int idx) const;

    int get_array_idx(const int v, const int u);
    std::map<int, std::vector<int>>& get_adjacent_nodes();
    std::map<std::pair<int, int>, int>& get_edge_to_array();
    void set_edge_to_deps(const int v, const int u, const std::string deps);
    std::string get_dep(const int v, const int u);
};
#endif
