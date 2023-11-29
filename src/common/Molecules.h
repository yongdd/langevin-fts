/*----------------------------------------------------------
* This class defines polymer molecules parameters
*-----------------------------------------------------------*/

#ifndef MOLECULES_H_
#define MOLECULES_H_

#include <string>
#include <vector>
#include <map>
#include "PropagatorCode.h"
#include "Propagators.h"
#include "Polymer.h"

class Molecules
{
private:
    std::string model_name; // "continuous": continuous standard Gaussian model
                            // "discrete": discrete bead-spring model
                            
    double ds;                          // contour step interval
    bool aggregate_propagator_computation; // compute multiple propagators using property of linearity of the diffusion equation.

    // dictionary{key:monomer_type, value:relative statistical_segment_length. (a_A/a_Ref)^2 or (a_B/a_Ref)^2, ...}
    std::map<std::string, double> bond_lengths;

    // Polymer types
    std::vector<Polymer> polymer_types;

    // set{key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u))}
    std::map<std::tuple<int, std::string, std::string>, ComputationBlock> essential_blocks;

    // dictionary{key:non-duplicated unique propagator_codes, value: ComputationEdge}
    std::map<std::string, ComputationEdge, ComparePropagatorKey> essential_propagator_codes; 

    // Add new key. if it already exists and 'new_n_segment' is larger than 'max_n_segment', update it.
    void update_essential_propagator_code(std::map<std::string, ComputationEdge, ComparePropagatorKey>& essential_propagator_codes, std::string new_key, int new_n_segment);

    // Superpose propagators
    std::map<std::string, ComputationBlock> superpose_propagator_common             (std::map<std::string, ComputationBlock> remaining_keys, int minimum_n_segment);
    std::map<std::string, ComputationBlock> superpose_propagator_of_continuous_chain(std::map<std::string, ComputationBlock> u_map);
    std::map<std::string, ComputationBlock> superpose_propagator_of_discrete_chain  (std::map<std::string, ComputationBlock> u_map);

public:

    Molecules(std::string model_name, double ds, std::map<std::string, double> bond_lengths, bool aggregate_propagator_computation);
    ~Molecules() {};

    std::string get_model_name() const;
    double get_ds() const;
    const std::map<std::string, double>& get_bond_lengths() const;
    bool is_using_propagator_aggregation() const;

    // Add new polymer
    // Mark some chain ends to set initial conditions of propagators when pseudo.compute_statistics() is invoked.
    // For instance, if chain_end_to_q_init[a] is set to b, 
    // Q_init[b] will be used as an initial condition of chain end 'a' in pseudo.compute_statistics().
    void add_polymer(
        double volume_fraction,
        std::vector<BlockInput> block_inputs,
        std::map<int, std::string> chain_end_to_q_init);

    // Add new polymer
    // All chain ends are free ends, e.g, q(r,0) = 1.
    void add_polymer(
        double volume_fraction,
        std::vector<BlockInput> block_inputs)
    {
        add_polymer(volume_fraction, block_inputs, {});
    }

    // Get polymers
    int get_n_polymer_types() const;
    Polymer& get_polymer(const int p);

    // Get information of essential propagators and blocks
    int get_n_essential_propagator_codes() const;
    std::map<std::string, ComputationEdge, ComparePropagatorKey>& get_essential_propagator_codes(); 
    ComputationEdge& get_essential_propagator_code(std::string key);
    std::map<std::tuple<int, std::string, std::string>, ComputationBlock>& get_essential_blocks(); 
    ComputationBlock& get_essential_block(std::tuple<int, std::string, std::string> key);

    // Display
    void display_propagators() const;
    void display_blocks() const;
    void display_sub_propagators() const;
};
#endif
