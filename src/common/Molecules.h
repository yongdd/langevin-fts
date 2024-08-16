/*----------------------------------------------------------
* This class defines polymer molecules parameters
*-----------------------------------------------------------*/

#ifndef MOLECULES_H_
#define MOLECULES_H_

#include <string>
#include <vector>
#include <map>

#include "PropagatorCode.h"
#include "Polymer.h"

class Molecules
{
private:
    std::string model_name; // "continuous": continuous standard Gaussian model
                            // "discrete": discrete bead-spring model
                            
    double ds;              // contour step interval

    // dictionary{key:monomer_type, value:relative statistical_segment_length. (a_A/a_Ref)^2 or (a_B/a_Ref)^2, ...}
    std::map<std::string, double> bond_lengths;

    // Polymer types
    std::vector<Polymer> polymer_types;

    // Solvent types  (volume fraction, monomer type)
    std::vector<std::tuple<double, std::string>> solvent_types;
public:

    Molecules(std::string model_name, double ds, std::map<std::string, double> bond_lengths);
    ~Molecules() {};

    std::string get_model_name() const;
    double get_ds() const;
    const std::map<std::string, double>& get_bond_lengths() const;

    // Add new polymer
    // Mark some chain ends to set initial conditions of propagators when solver.compute_propagators() is invoked.
    // For instance, if chain_end_to_q_init[a] is set to b, 
    // q_init[b] will be used as an initial condition of chain end 'a' in solver.compute_propagators().
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

    // Add solvent
    void add_solvent(double volume_fraction, std::string monomer_type);

    // Get polymers
    int get_n_polymer_types() const;
    Polymer& get_polymer(const int p);

    // Get solvents
    int get_n_solvent_types() const;
    std::tuple<double, std::string> get_solvent(const int s);
};
#endif
