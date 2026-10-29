/**
 * @file Molecules.cpp
 * @brief Implementation of Molecules container class.
 *
 * Manages collections of polymer and solvent species for field theory
 * simulations. Stores chain model parameters (continuous/discrete),
 * contour step size, and statistical segment lengths.
 *
 * **Chain Models:**
 *
 * - Continuous: Modified diffusion equation (Gaussian chain)
 * - Discrete: Integral equation (bead-spring model)
 *
 * **Species Management:**
 *
 * - Polymers: Added via add_polymer() with block architecture
 * - Solvents: Added via add_solvent() with monomer type
 *
 * @see Polymer for individual chain architecture
 * @see PropagatorComputation for field theory calculations
 */

#include <iostream>
#include <cctype>
#include <cmath>
#include <numbers>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "Molecules.h"
#include "PropagatorCode.h"
#include "Exception.h"
#include "ValidationUtils.h"

/**
 * @brief Construct Molecules container.
 *
 * Validates chain model name and stores simulation parameters.
 *
 * @param model_name    Chain model: "continuous" or "discrete"
 * @param ds            Contour step size (1/N_Ref typically)
 * @param bond_lengths  Statistical segment lengths by monomer type
 *
 * @throws Exception if model_name is invalid
 */
Molecules::Molecules(
    std::string model_name, double ds, std::map<std::string, double> bond_lengths)
{
    // Transform into lower cases
    std::string model_lower = validation::to_lower(model_name);

    // Check chain model
    if (model_lower != "continuous" && model_lower != "discrete")
    {
        throw_with_line_number(model_name + " is an invalid chain model. This must be 'continuous' or 'discrete'.")
    }

    // Save variables
    this->ds = ds;
    this->bond_lengths = bond_lengths;
    this->model_name = model_lower;

    // Initialize contour length mapping with global ds
    this->contour_length_mapping = ContourLengthMapping(ds);
}

/**
 * @brief Add a polymer species to the mixture.
 *
 * Creates a new Polymer object with the given architecture and appends
 * it to the polymer_types vector.
 *
 * @param volume_fraction     Volume fraction of this polymer species
 * @param block_inputs        Block definitions [monomer, length, v, u]
 * @param chain_end_to_q_init Custom initial conditions at chain ends
 *
 * @see Polymer::Polymer for block validation
 */
void Molecules::add_polymer(
    double volume_fraction,
    std::vector<BlockInput> block_inputs,
    std::map<int, std::string> chain_end_to_q_init)
{
    if (contour_length_mapping.finalized())
        throw_with_line_number("Cannot add a polymer after a propagator computation has been created from these molecules.");

    // For discrete chains, validate that contour_length is an integer multiple of ds
    if (model_name == "discrete")
    {
        for (auto& block : block_inputs)
        {
            // The discrete solver uses one global contour step. Accepted
            // roundoff must not register a second, slightly different local ds.
            block.contour_length = ContourLengthMapping::discrete_contour_length(block.contour_length, ds);
        }
    }

    // Validate the polymer before registering anything, so that a rejected
    // polymer leaves no contour lengths in the mapping
    Polymer polymer(ds, bond_lengths, volume_fraction, block_inputs, chain_end_to_q_init);

    // Register block contour lengths in the mapping
    for (const auto& block : block_inputs)
    {
        contour_length_mapping.add_block(block.contour_length);
    }

    // Add new polymer type
    polymer_types.push_back(std::move(polymer));
}

/**
 * @brief Add a solvent species to the mixture.
 *
 * Solvents are point particles (no chain contour) identified by
 * their monomer type for field interactions.
 *
 * @param volume_fraction Volume fraction of this solvent
 * @param monomer_type    Monomer type for χ interactions
 */
void Molecules::add_solvent(
    double volume_fraction, std::string monomer_type)
{
    // A propagator computation sizes its solvent arrays when it is created
    if (contour_length_mapping.finalized())
        throw_with_line_number("Cannot add a solvent after a propagator computation has been created from these molecules.");
    validation::require_in_range(volume_fraction, 0.0, 1.0, "volume_fraction of the solvent");
    // As for polymer blocks: the solvers keep a Boltzmann factor for each type of bond_lengths only
    validation::require_string_key(bond_lengths, monomer_type, "bond_lengths");

    solvent_types.push_back(std::make_tuple(volume_fraction, monomer_type));
}
std::string Molecules::get_model_name() const
{
    return model_name;
}
double Molecules::get_global_ds() const
{
    return ds;
}
int Molecules::get_n_polymer_types() const
{
    return polymer_types.size();
}
Polymer& Molecules::get_polymer(const int p)
{
    if (p < 0 || p >= static_cast<int>(polymer_types.size()))
        throw_with_line_number("Polymer index (" + std::to_string(p) + ") must be in range [0, " +
            std::to_string(static_cast<int>(polymer_types.size()) - 1) + "]");
    return polymer_types[p];
}
const std::map<std::string, double>& Molecules::get_bond_lengths() const
{
    return bond_lengths;
}
int Molecules::get_n_solvent_types() const
{
    return solvent_types.size();
}
std::tuple<double, std::string> Molecules::get_solvent(const int s) const
{
    if (s < 0 || s >= static_cast<int>(solvent_types.size()))
        throw_with_line_number("Solvent index (" + std::to_string(s) + ") must be in range [0, " +
            std::to_string(static_cast<int>(solvent_types.size()) - 1) + "]");
    return solvent_types[s];
}

/**
 * @brief Finalize the contour length mapping.
 *
 * Builds the integer index mappings for all unique contour lengths
 * and local Δs values collected from the added polymers.
 * Also regenerates propagator keys for all polymers using the length indices.
 */
void Molecules::finalize_contour_length_mapping()
{
    if (!contour_length_mapping.finalized())
    {
        contour_length_mapping.finalize();

        // Regenerate propagator keys for all polymers using length indices
        for (auto& polymer : polymer_types)
        {
            polymer.regenerate_propagator_keys(contour_length_mapping);
        }
    }
}

/**
 * @brief Get the contour length mapping.
 */
ContourLengthMapping& Molecules::get_contour_length_mapping()
{
    return contour_length_mapping;
}

/**
 * @brief Get the contour length mapping (const version).
 */
const ContourLengthMapping& Molecules::get_contour_length_mapping() const
{
    return contour_length_mapping;
}

/**
 * @brief Display ASCII art architecture diagrams for all polymers.
 *
 * Prints a single title, then each polymer's diagram with its index,
 * followed by a legend explaining the notation.
 */
void Molecules::display_architectures() const
{
    if (polymer_types.empty())
    {
        return;
    }

    std::cout << "=== Polymer Architectures ===" << std::endl;
    for (size_t p = 0; p < polymer_types.size(); p++)
    {
        std::cout << "[" << p << "]" << std::endl;
        polymer_types[p].print_architecture_diagram();
    }
    std::cout << "Legend: (n)=vertex, Type[length]=block" << std::endl;
}
