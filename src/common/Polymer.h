/*----------------------------------------------------------
* This class defines single polymer chain parameters
*-----------------------------------------------------------*/

#ifndef POLYMER_H_
#define POLYMER_H_

#include <string>
#include <vector>
#include <map>

// Block information as an input
struct BlockInput{
    std::string monomer_type;   // monomer_type of block, e.g., "A" or "B", ...
    double contour_length;      // relative length of each block, e.g., 1.0, 0.4, 0.3, ...
    int v;                      // starting vertex (or node)
    int u;                      // ending vertex (or node)
};

// Block information as an internal data
struct Block{
    std::string monomer_type;   // monomer_type of block, e.g., "A" or "B", ...
    int n_segment;              // The number of segment, e.g. 20, or 30 ...
    double contour_length;      // relative length of each block, e.g., 1.0, 0.4, 0.3, ...
    int v;                      // starting vertex (or node)
    int u;                      // ending vertex (or node)
};

class Polymer
{
private:
    double alpha;            // sum of contour lengths
    double volume_fraction;  // volume_fraction

    std::vector<Block> blocks;  // information of blocks, which contains 'monomer_type', 'n_segments', '
                                // bond_length_sq', 'contour_length', and 'vertices'.

    std::map<int, std::vector<int>> adjacent_nodes;                     // adjacent nodes
    std::map<std::pair<int, int>, int>         edge_to_block_index;     // array index for each edge
    std::map<std::pair<int, int>, std::string> edge_to_propagator_key;  // propagator code as a text

    // Grafting point.
    // For instance, 'chain_end_to_q_init[a] = b' means that
    // The initial condition of chain end vertex 'a' will be given as 'initial[b]' in solver.compute_statistics()
    std::map<int, std::string> chain_end_to_q_init;

    // Convert a number to a string of alphabets
    std::string generateString(int number) {
        std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::string result = "";

        number++;
        while (number > 0) {
            int remainder = (number - 1) % 26;
            result = alphabet[remainder] + result;
            number = (number - 1) / 26;
        }

        return result;
    }

public:
    Polymer(
        int index, 
        double ds, std::map<std::string, double> bond_lengths, 
        double volume_fraction, 
        std::vector<BlockInput> block_inputs,
        std::map<int, std::string> chain_end_to_q_init={});
    ~Polymer() {};

    double get_alpha() const;
    double get_volume_fraction() const;

    int get_n_blocks() const;
    std::vector<Block>& get_blocks();
    struct Block& get_block(const int v, const int u);

    int get_n_segment_total() const;
    int get_n_segment(const int idx) const;

    int get_block_index_from_edge(const int v, const int u);
    std::map<int, std::vector<int>>& get_adjacent_nodes();
    std::map<std::pair<int, int>, int>& get_block_indexes();
    void set_propagator_key(const std::string deps, const int v, const int u);
    std::string get_propagator_key(const int v, const int u);
};
#endif
