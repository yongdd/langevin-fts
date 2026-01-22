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
 * - Discrete: Integral equation (freely-jointed chain)
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
        throw_with_line_number(model_name + " is an invalid chain model. This must be 'Continuous' or 'Discrete'.")
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
    // For discrete chains, validate that contour_length is an integer multiple of ds
    if (model_name == "discrete")
    {
        for (const auto& block : block_inputs)
        {
            double ratio = block.contour_length / ds;
            int n_segment = static_cast<int>(std::round(ratio));
            double expected_length = n_segment * ds;

            if (std::abs(block.contour_length - expected_length) > 1e-9)
            {
                throw_with_line_number(
                    "For discrete chain model, block contour_length (" +
                    std::to_string(block.contour_length) +
                    ") must be an integer multiple of ds (" +
                    std::to_string(ds) + "). " +
                    "Closest valid value: " + std::to_string(expected_length));
            }
        }
    }

    // Register block contour lengths in the mapping
    for (const auto& block : block_inputs)
    {
        contour_length_mapping.add_block(block.contour_length);
    }

    // Add new polymer type
    polymer_types.push_back(Polymer(ds, bond_lengths,
        volume_fraction, block_inputs, chain_end_to_q_init));
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
    // Add new polymer type
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