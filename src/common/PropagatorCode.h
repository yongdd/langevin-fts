/*----------------------------------------------------------
* This class contains functions for propagator code
*-----------------------------------------------------------*/

#ifndef PROPAGATOR_CODE_H_
#define PROPAGATOR_CODE_H_

#include <string>
#include <vector>
#include <map>
#include "Polymer.h"

class PropagatorCode
{
private:
    // This method is invoked by generate_code()
    static std::pair<std::string, int> generate_edge_code(
        std::map<std::pair<int, int>, std::pair<std::string, int>>& memory,
        std::vector<Block>& blocks,
        std::map<int, std::vector<int>>& adjacent_nodes,
        std::map<std::pair<int, int>, int>& edge_to_block_index,
        std::map<int, std::string>& chain_end_to_q_init,
        int in_node, int out_node);
public:

    // PropagatorCode();
    // ~PropagatorCode() {};

    // Get all propagator codes in given polymer
    // This method is implemented using memoization (top-down dynamic programming approach).
    static std::vector<std::tuple<int, int, std::string>> generate_codes(Polymer& pc, std::map<int, std::string>& chain_end_to_q_init);

    // Get information from key
    static std::vector<std::tuple<std::string, int, int>> get_deps_from_key(std::string key);
    static std::string remove_monomer_type_from_key(std::string key);
    static std::string get_monomer_type_from_key(std::string key);
    static std::string get_q_input_idx_from_key(std::string key);
    static int get_height_from_key(std::string key);
    
};
#endif
