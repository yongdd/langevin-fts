/**
 * @file Polymer.h
 * @brief Defines polymer chain architecture and topology for field theory simulations.
 *
 * This header provides the Polymer class and related structures (BlockInput, Block)
 * for representing arbitrary acyclic branched polymer architectures. The class
 * supports linear, star, comb, dendritic, bottle-brush, and other branched
 * topologies through a graph-based representation.
 *
 * **Polymer Representation:**
 *
 * A polymer is represented as an undirected graph where:
 * - Vertices (nodes) represent junction points or chain ends
 * - Edges represent polymer blocks connecting vertices
 * - Each edge has a monomer type and contour length
 *
 * For example, an AB diblock copolymer has 3 vertices and 2 edges:
 * ```
 *   0 ---A--- 1 ---B--- 2
 * ```
 *
 * A 3-arm star polymer has 4 vertices and 3 edges:
 * ```
 *       1
 *       |A
 *   2-B-0-C-3
 * ```
 *
 * @note The polymer graph must be acyclic (a tree structure).
 *
 * @see Molecules for managing collections of polymer types
 * @see PropagatorCode for generating propagator computation codes from topology
 *
 * @example
 * @code
 * // Define an AB diblock copolymer with fA = 0.3
 * std::vector<BlockInput> blocks = {
 *     {"A", 0.3, 0, 1},  // A block: length 0.3, from vertex 0 to 1
 *     {"B", 0.7, 1, 2}   // B block: length 0.7, from vertex 1 to 2
 * };
 *
 * std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
 * double ds = 1.0 / 100;  // Contour step
 * double vol_frac = 1.0;  // Volume fraction
 *
 * Polymer diblock(ds, bond_lengths, vol_frac, blocks);
 *
 * // Define a 3-arm ABC star polymer
 * std::vector<BlockInput> star_blocks = {
 *     {"A", 0.33, 0, 1},  // A arm
 *     {"B", 0.33, 0, 2},  // B arm
 *     {"C", 0.34, 0, 3}   // C arm
 * };
 * Polymer star(ds, bond_lengths, 1.0, star_blocks);
 * @endcode
 */

#ifndef POLYMER_H_
#define POLYMER_H_

#include <string>
#include <vector>
#include <map>

// Forward declaration
class ContourLengthMapping;

/**
 * @struct BlockInput
 * @brief Input specification for a polymer block (user-facing structure).
 *
 * This structure is used when defining polymer architectures. It specifies
 * the monomer type, contour length, and connectivity (which vertices the
 * block connects).
 *
 * @example
 * @code
 * // A-block connecting vertex 0 to vertex 1 with length 0.5
 * BlockInput block_A = {"A", 0.5, 0, 1};
 *
 * // B-block connecting vertex 1 to vertex 2 with length 0.5
 * BlockInput block_B = {"B", 0.5, 1, 2};
 * @endcode
 */
struct BlockInput{
    std::string monomer_type;   ///< Monomer type label, e.g., "A", "B", "C"
    double contour_length;      ///< Relative contour length (fraction of reference chain)
    int v;                      ///< Starting vertex (node) index
    int u;                      ///< Ending vertex (node) index
};

/**
 * @struct Block
 * @brief Internal representation of a polymer block with discretization info.
 *
 * This structure extends BlockInput with discretization information (n_segment)
 * computed from the contour length and step size ds. Used internally by the
 * propagator solver.
 *
 * @note n_segment = round(contour_length / ds), with special handling for
 *       continuous vs discrete chain models.
 */
struct Block{
    std::string monomer_type;   ///< Monomer type label, e.g., "A", "B", "C"
    int n_segment;              ///< Number of contour segments (steps) in this block
    double contour_length;      ///< Relative contour length (N_block / N_Ref)
    int v;                      ///< Starting vertex (node) index
    int u;                      ///< Ending vertex (node) index
};

/**
 * @class Polymer
 * @brief Represents a single polymer chain type with arbitrary branched architecture.
 *
 * The Polymer class stores the topology and parameters of a polymer chain type.
 * It uses a graph representation where vertices are junction points or chain ends,
 * and edges are polymer blocks. The class provides methods to query topology
 * information and generates propagator computation codes for efficient solver
 * scheduling.
 *
 * **Key Concepts:**
 *
 * - **Vertices (Nodes)**: Junction points (degree > 2) or chain ends (degree = 1)
 * - **Edges (Blocks)**: Polymer segments connecting two vertices
 * - **Contour Length (alpha)**: Total chain length in units of N_Ref
 * - **Volume Fraction**: Fraction of system volume occupied by this polymer type
 *
 * **Propagator Computation:**
 *
 * The class generates "propagator codes" that encode dependencies between
 * propagators for efficient dynamic programming computation. This is described
 * in J. Chem. Theory Comput. 2025, 21, 3676.
 *
 * **Grafting:**
 *
 * For grafted chains, chain_end_to_q_init maps vertex indices to initial
 * condition labels. This allows specifying q(r,0) from external sources
 * (e.g., grafting density on a surface).
 *
 * @see BlockInput, Block for block specification structures
 * @see Molecules for managing multiple polymer types
 * @see PropagatorCode for code generation algorithm
 */
class Polymer
{
private:
    double alpha;            ///< Total contour length (sum of all block lengths)
    double volume_fraction;  ///< Volume fraction of this polymer type in the blend

    /**
     * @brief Vector of all blocks in the polymer.
     *
     * Contains discretization info (n_segment) in addition to input parameters.
     */
    std::vector<Block> blocks;

    /**
     * @brief Adjacency list representation of polymer graph.
     *
     * Maps each vertex to a list of adjacent vertices.
     * adjacent_nodes[v] = {u1, u2, ...} means edges (v,u1), (v,u2), ... exist.
     */
    std::map<int, std::vector<int>> adjacent_nodes;

    /**
     * @brief Maps edge (v,u) to block index in the blocks vector.
     *
     * Both (v,u) and (u,v) map to the same block index.
     */
    std::map<std::pair<int, int>, int> edge_to_block_index;

    /**
     * @brief Maps edge (v,u) to its propagator computation code.
     *
     * The code encodes dependencies for dynamic programming optimization.
     */
    std::map<std::pair<int, int>, std::string> edge_to_propagator_key;

    /**
     * @brief Maps chain end vertices to initial condition labels.
     *
     * For grafted chains: chain_end_to_q_init[a] = "label" means that
     * the propagator starting from vertex 'a' uses q_init["label"]
     * as its initial condition instead of q(r,0) = 1.
     *
     * @example
     * @code
     * // Grafted chain with end vertex 0 on surface
     * std::map<int, std::string> grafting = {{0, "surface"}};
     * // Then in compute_propagators(), pass q_init["surface"] = grafting_density
     * @endcode
     */
    std::map<int, std::string> chain_end_to_q_init;

public:
    /**
     * @brief Construct a Polymer from block specifications.
     *
     * @param ds              Contour step size (typically 1/N_Ref)
     * @param bond_lengths    Statistical segment lengths squared: {type: (a/a_Ref)^2}
     * @param volume_fraction Volume fraction of this polymer type
     * @param block_inputs    Vector of BlockInput defining topology
     * @param chain_end_to_q_init Optional: map chain ends to initial condition labels
     *
     * @throws Exception if topology is invalid (cycles, disconnected, etc.)
     *
     * @example
     * @code
     * // AB diblock with fA = 0.3
     * std::vector<BlockInput> blocks = {{"A", 0.3, 0, 1}, {"B", 0.7, 1, 2}};
     * std::map<std::string, double> bonds = {{"A", 1.0}, {"B", 1.0}};
     * Polymer p(1.0/100, bonds, 1.0, blocks);
     *
     * // Grafted chain on surface
     * std::map<int, std::string> graft = {{0, "surface"}};
     * Polymer grafted(1.0/100, bonds, 1.0, blocks, graft);
     * @endcode
     */
    Polymer(
        double ds, std::map<std::string, double> bond_lengths,
        double volume_fraction,
        std::vector<BlockInput> block_inputs,
        std::map<int, std::string> chain_end_to_q_init={});

    /**
     * @brief Destructor.
     */
    ~Polymer() {};

    /**
     * @brief Get total contour length.
     * @return Sum of all block contour lengths (alpha = N/N_Ref)
     */
    double get_alpha() const;

    /**
     * @brief Get volume fraction of this polymer type.
     * @return Volume fraction (0 to 1)
     */
    double get_volume_fraction() const;

    /**
     * @brief Get number of blocks in polymer.
     * @return Number of edges in polymer graph
     */
    int get_n_blocks() const;

    /**
     * @brief Get reference to all blocks.
     * @return Vector of Block structures
     */
    std::vector<Block>& get_blocks();

    /**
     * @brief Get block connecting vertices v and u.
     * @param v First vertex index
     * @param u Second vertex index
     * @return Reference to Block structure
     * @throws Exception if edge (v,u) doesn't exist
     */
    struct Block& get_block(const int v, const int u);

    /**
     * @brief Get total number of contour segments.
     * @return Sum of n_segment over all blocks
     */
    int get_n_segment_total() const;

    /**
     * @brief Get number of segments in a specific block.
     * @param idx Block index (0 to n_blocks-1)
     * @return n_segment for the specified block
     */
    int get_n_segment(const int idx) const;

    /**
     * @brief Get block index from edge vertices.
     * @param v First vertex
     * @param u Second vertex
     * @return Index into blocks vector
     */
    int get_block_index_from_edge(const int v, const int u) const;

    /**
     * @brief Get adjacency list representation.
     * @return Map from vertex to list of adjacent vertices
     */
    std::map<int, std::vector<int>>& get_adjacent_nodes();

    /**
     * @brief Get edge to block index mapping.
     * @return Map from (v,u) pair to block index
     */
    std::map<std::pair<int, int>, int>& get_block_indexes();

    /**
     * @brief Set propagator computation code for an edge.
     * @param deps Dependency code string
     * @param v First vertex
     * @param u Second vertex
     */
    void set_propagator_key(const std::string deps, const int v, const int u);

    /**
     * @brief Get propagator computation code for an edge.
     * @param v First vertex
     * @param u Second vertex
     * @return Propagator code string encoding dependencies
     */
    std::string get_propagator_key(const int v, const int u) const;

    /**
     * @brief Get the chain_end_to_q_init mapping.
     * @return Reference to the chain end to q_init mapping
     */
    std::map<int, std::string>& get_chain_end_to_q_init();

    /**
     * @brief Regenerate propagator keys using contour length mapping.
     *
     * Replaces the propagator keys with new ones that use length indices
     * instead of segment counts. This enables support for floating-point
     * block lengths.
     *
     * @param mapping Contour length mapping (must be finalized)
     */
    void regenerate_propagator_keys(const ContourLengthMapping& mapping);

    /**
     * @brief Display polymer architecture as ASCII art diagram.
     *
     * Prints a visual representation of the polymer topology showing
     * vertices (junction points and chain ends) connected by blocks
     * with their monomer types and contour lengths.
     *
     * @param polymer_id Optional polymer index to include in the title.
     *                   If >= 0, displays "=== Polymer N ==="
     *                   If < 0 (default), displays "=== Polymer ==="
     * @param show_legend If true, prints the legend explaining symbols (default: false).
     * @param show_title If true (default), prints the title header.
     *
     * For linear chains, outputs a single-line format:
     * ```
     * (0)--A[0.30]--(1)--B[0.70]--(2)
     * ```
     *
     * For branched polymers, uses tree format:
     * ```
     * (0)
     *  +--A[0.33]--(1)
     *  +--B[0.33]--(2)
     *  +--C[0.34]--(3)
     * ```
     */
    void display_architecture(int polymer_id = -1, bool show_legend = false, bool show_title = true) const;
};
#endif
