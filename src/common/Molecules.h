/**
 * @file Molecules.h
 * @brief Container class for managing polymer and solvent species in a simulation.
 *
 * This header provides the Molecules class which serves as a container for all
 * molecular species (polymers and solvents) in a field theory simulation. It
 * manages the chain model type (continuous or discrete), contour step size,
 * and statistical segment lengths for all monomer types.
 *
 * **Supported Species:**
 *
 * 1. **Polymers**: Arbitrary branched architectures defined by Polymer class
 * 2. **Solvents**: Point-like molecules with a single monomer type
 *
 * **Chain Models:**
 *
 * - **Continuous**: Standard Gaussian chain model with continuous contour.
 *   Uses pseudo-spectral method with RQM4 (4th-order Richardson extrapolation).
 * - **Discrete**: Bead-spring model with discrete segments.
 *   More accurate for short chains or when segment-level detail matters.
 *
 * @see Polymer for polymer chain definitions
 * @see PropagatorComputation for solving chain propagator equations
 *
 * @example
 * @code
 * // Create a binary AB diblock blend with solvent
 * std::map<std::string, double> bonds = {{"A", 1.0}, {"B", 1.0}, {"S", 1.0}};
 * Molecules molecules("continuous", 1.0/100, bonds);
 *
 * // Add AB diblock copolymer (80% by volume)
 * std::vector<BlockInput> diblock = {{"A", 0.3, 0, 1}, {"B", 0.7, 1, 2}};
 * molecules.add_polymer(0.8, diblock);
 *
 * // Add solvent (20% by volume)
 * molecules.add_solvent(0.2, "S");
 *
 * // Query species
 * int n_poly = molecules.get_n_polymer_types();   // 1
 * int n_solv = molecules.get_n_solvent_types();   // 1
 * double ds = molecules.get_ds();                  // 0.01
 * @endcode
 */

#ifndef MOLECULES_H_
#define MOLECULES_H_

#include <string>
#include <vector>
#include <map>

#include "PropagatorCode.h"
#include "Polymer.h"
#include "ContourLengthMapping.h"

/**
 * @class Molecules
 * @brief Container for all polymer and solvent species in a simulation.
 *
 * The Molecules class aggregates all molecular species and their parameters.
 * It is passed to the propagator solver which uses this information to
 * compute chain statistics for all species.
 *
 * **Key Parameters:**
 *
 * - **model_name**: "continuous" or "discrete" chain model
 * - **ds**: Contour step size for propagator integration
 * - **bond_lengths**: Map of monomer types to squared segment lengths
 *
 * **Volume Fractions:**
 *
 * Volume fractions of all species must sum to 1.0 for incompressible systems.
 * The code does not enforce this; it's the user's responsibility.
 *
 * **Bond Lengths:**
 *
 * The bond_lengths map stores (a_i/a_Ref)^2 for each monomer type i.
 * These affect both the Laplacian operator in the diffusion equation and
 * stress calculations for box relaxation.
 *
 * @see Polymer for individual polymer chain definitions
 * @see AbstractFactory::create_molecules_information for factory creation
 */
class Molecules
{
private:
    /**
     * @brief Chain model name.
     *
     * - "continuous": Continuous Gaussian chain (pseudo-spectral method)
     * - "discrete": Discrete bead-spring chain
     */
    std::string model_name;

    /**
     * @brief Contour step size for propagator integration.
     *
     * Typical value: ds = 1/N_Ref where N_Ref is the reference chain length.
     * Smaller ds gives higher accuracy but more computational cost.
     */
    double ds;

    /**
     * @brief Statistical segment lengths squared for each monomer type.
     *
     * Maps monomer type label (e.g., "A", "B") to (a_i/a_Ref)^2 where
     * a_i is the statistical segment length and a_Ref is the reference.
     *
     * @example
     * @code
     * // Symmetric blend: both have same segment length
     * bond_lengths = {{"A", 1.0}, {"B", 1.0}};
     *
     * // Asymmetric: B segments are 1.2x longer
     * bond_lengths = {{"A", 1.0}, {"B", 1.44}};  // (1.2)^2 = 1.44
     * @endcode
     */
    std::map<std::string, double> bond_lengths;

    /**
     * @brief Vector of polymer types in the simulation.
     *
     * Each Polymer object defines one polymer species with its own
     * architecture and volume fraction.
     */
    std::vector<Polymer> polymer_types;

    /**
     * @brief Vector of solvent types (volume fraction, monomer type).
     *
     * Solvents are point-like molecules characterized only by their
     * monomer type (for interactions) and volume fraction.
     */
    std::vector<std::tuple<double, std::string>> solvent_types;

    /**
     * @brief Mapping from floating-point contour lengths and local Δs to integers.
     *
     * This mapping enables reliable comparison of propagator keys when block
     * lengths are arbitrary floating-point numbers. It assigns unique integer
     * indices to each distinct contour_length and local_ds value.
     */
    ContourLengthMapping contour_length_mapping;

public:
    /**
     * @brief Construct a Molecules container.
     *
     * @param model_name   Chain model: "continuous" or "discrete"
     * @param ds           Contour step size (typically 1/N_Ref)
     * @param bond_lengths Map of monomer types to (a/a_Ref)^2 values
     *
     * @throws Exception if model_name is invalid
     *
     * @example
     * @code
     * std::map<std::string, double> bonds = {{"A", 1.0}, {"B", 1.0}};
     * Molecules mols("continuous", 1.0/100, bonds);
     * @endcode
     */
    Molecules(std::string model_name, double ds, std::map<std::string, double> bond_lengths);

    /**
     * @brief Destructor.
     */
    ~Molecules() {};

    /**
     * @brief Get chain model name.
     * @return "continuous" or "discrete"
     */
    std::string get_model_name() const;

    /**
     * @brief Get contour step size.
     * @return ds value
     */
    double get_ds() const;

    /**
     * @brief Get bond lengths map.
     * @return Const reference to {monomer_type: (a/a_Ref)^2} map
     */
    const std::map<std::string, double>& get_bond_lengths() const;

    /**
     * @brief Add a new polymer type with grafting specification.
     *
     * @param volume_fraction Volume fraction of this polymer type
     * @param block_inputs    Vector of BlockInput defining chain topology
     * @param chain_end_to_q_init Map chain ends to initial condition labels for grafting
     *
     * @note For grafted chains, chain_end_to_q_init[v] = "label" means vertex v
     *       uses q_init["label"] as initial condition instead of q(r,0)=1.
     *
     * @example
     * @code
     * // Grafted chain on surface
     * std::vector<BlockInput> chain = {{"A", 1.0, 0, 1}};
     * std::map<int, std::string> graft = {{0, "surface"}};
     * molecules.add_polymer(0.5, chain, graft);
     * @endcode
     */
    void add_polymer(
        double volume_fraction,
        std::vector<BlockInput> block_inputs,
        std::map<int, std::string> chain_end_to_q_init);

    /**
     * @brief Add a new polymer type with free ends.
     *
     * Convenience overload where all chain ends have q(r,0) = 1.
     *
     * @param volume_fraction Volume fraction of this polymer type
     * @param block_inputs    Vector of BlockInput defining chain topology
     *
     * @example
     * @code
     * // Free AB diblock
     * std::vector<BlockInput> diblock = {{"A", 0.3, 0, 1}, {"B", 0.7, 1, 2}};
     * molecules.add_polymer(1.0, diblock);
     * @endcode
     */
    void add_polymer(
        double volume_fraction,
        std::vector<BlockInput> block_inputs)
    {
        add_polymer(volume_fraction, block_inputs, {});
    }

    /**
     * @brief Add a solvent species.
     *
     * Solvents are point-like molecules with a single monomer type.
     * They contribute to the incompressibility constraint and interact
     * with polymer segments via chi parameters.
     *
     * @param volume_fraction Volume fraction of solvent
     * @param monomer_type    Monomer type label (e.g., "S")
     *
     * @example
     * @code
     * molecules.add_solvent(0.2, "S");  // 20% solvent of type "S"
     * @endcode
     */
    void add_solvent(double volume_fraction, std::string monomer_type);

    /**
     * @brief Get number of polymer types.
     * @return Count of distinct polymer species
     */
    int get_n_polymer_types() const;

    /**
     * @brief Get polymer by index.
     * @param p Polymer index (0 to n_polymer_types-1)
     * @return Reference to Polymer object
     */
    Polymer& get_polymer(const int p);

    /**
     * @brief Display ASCII art architecture diagrams for all polymers.
     *
     * Iterates through all polymer types and displays their topology
     * using the Polymer::display_architecture() method. Each polymer
     * is labeled with its index.
     *
     * Example output:
     * ```
     * === Polymer 0 Architecture ===
     * (0)--A[0.30]--(1)--B[0.70]--(2)
     * Legend: (n)=vertex index, X=monomer type, [L]=contour length
     * === Polymer 1 Architecture ===
     * (0)--A[0.50]--(1)--B[0.50]--(2)
     * Legend: (n)=vertex index, X=monomer type, [L]=contour length
     * ```
     */
    void display_architectures() const;

    /**
     * @brief Get number of solvent types.
     * @return Count of distinct solvent species
     */
    int get_n_solvent_types() const;

    /**
     * @brief Get solvent by index.
     * @param s Solvent index (0 to n_solvent_types-1)
     * @return Tuple of (volume_fraction, monomer_type)
     */
    std::tuple<double, std::string> get_solvent(const int s) const;

    /**
     * @brief Finalize the contour length mapping.
     *
     * Must be called after all polymers have been added and before
     * using the mapping. Collects all unique contour lengths from
     * all polymers and builds the integer index mappings.
     *
     * @note This is called automatically by PropagatorComputationOptimizer,
     *       but can be called explicitly if mapping is needed earlier.
     */
    void finalize_contour_length_mapping();

    /**
     * @brief Get the contour length mapping.
     * @return Reference to ContourLengthMapping object
     */
    ContourLengthMapping& get_contour_length_mapping();

    /**
     * @brief Get the contour length mapping (const version).
     * @return Const reference to ContourLengthMapping object
     */
    const ContourLengthMapping& get_contour_length_mapping() const;
};
#endif
