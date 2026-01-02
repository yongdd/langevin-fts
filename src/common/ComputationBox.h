/**
 * @file ComputationBox.h
 * @brief Defines the computational domain (grid and box) for polymer simulations.
 *
 * This header provides the ComputationBox class which manages the discrete spatial
 * grid used in polymer field theory simulations. It defines grid dimensions, box
 * lengths, boundary conditions, and provides methods for computing integrals and
 * inner products over the simulation domain.
 *
 * The class supports 1D, 2D, and 3D simulations with various boundary conditions:
 * - Periodic: Standard periodic boundary conditions (for pseudo-spectral method)
 * - Reflecting: Zero-flux Neumann boundary conditions (for real-space method)
 * - Absorbing: Zero-value Dirichlet boundary conditions (for real-space method)
 *
 * @note All lengths are in units of a_Ref * N_Ref^(1/2), where a_Ref is the
 *       reference statistical segment length and N_Ref is the reference
 *       polymerization index.
 *
 * @see CpuComputationBox, CudaComputationBox for platform-specific implementations
 *
 * @example
 * @code
 * // Create a 3D computation box with 32^3 grid points
 * std::vector<int> nx = {32, 32, 32};
 * std::vector<double> lx = {4.0, 4.0, 4.0};
 * std::vector<std::string> bc = {"periodic", "periodic", "periodic",
 *                                 "periodic", "periodic", "periodic"};
 *
 * ComputationBox<double>* cb = factory->create_computation_box(nx, lx, bc);
 *
 * // Compute volume integral of a field
 * double integral_phi = cb->integral(phi);
 *
 * // Compute inner product <g|h> = integral(g * h)
 * double ip = cb->inner_product(g, h);
 * @endcode
 */
#ifndef COMPUTATION_BOX_H_
#define COMPUTATION_BOX_H_

#include <array>
#include <vector>
#include <cassert>
#include <complex>

#include "Exception.h"

/**
 * @enum BoundaryCondition
 * @brief Types of boundary conditions supported by the simulation.
 *
 * Different solvers support different boundary conditions:
 * - Pseudo-spectral method: Only PERIODIC is supported
 * - Real-space method: All three types are supported
 */
enum class BoundaryCondition
{
    PERIODIC,    ///< Periodic BC: f(0) = f(L), f'(0) = f'(L). Used with FFT-based solvers.
    REFLECTING,  ///< Reflecting (Neumann) BC: df/dn = 0 at boundary. Models impenetrable walls.
    ABSORBING,   ///< Absorbing (Dirichlet) BC: f = 0 at boundary. Models chain-absorbing surfaces.
};

/**
 * @class ComputationBox
 * @brief Manages the computational domain grid and provides integration methods.
 *
 * This template class defines the spatial discretization for polymer field theory
 * simulations. It stores grid dimensions (nx), box lengths (lx), and boundary
 * conditions. The class provides methods for computing volume integrals and
 * inner products, which are fundamental operations in SCFT and L-FTS calculations.
 *
 * @tparam T Numeric type for field values (double or std::complex<double>)
 *
 * **Grid Layout:**
 *
 * For a 3D box with dimensions nx = [Nx, Ny, Nz]:
 * - Total grid points: M = Nx * Ny * Nz
 * - Grid spacing: dx[i] = lx[i] / nx[i]
 * - Volume element: dV = dx[0] * dx[1] * dx[2]
 *
 * **Memory Layout:**
 *
 * Fields are stored as 1D arrays in row-major order (C-style):
 * index = iz + Nz * (iy + Ny * ix)
 *
 * **Units:**
 *
 * - Length unit: a_Ref * N_Ref^(1/2)
 * - Volume unit: (a_Ref * N_Ref^(1/2))^dim
 *
 * @note This is a base class. Use CpuComputationBox or CudaComputationBox
 *       for actual computations.
 */
template <typename T>
class ComputationBox
{
protected:
    int dim;                      ///< Spatial dimension (1, 2, or 3)
    std::vector<int> nx;          ///< Number of grid points in each direction [Nx, Ny, Nz]
    std::vector<double> lx;       ///< Box lengths in each direction [Lx, Ly, Lz] (units: a_Ref*N^0.5)
    std::vector<double> dx;       ///< Grid spacing in each direction [dx, dy, dz] = lx/nx
    int total_grid;               ///< Total number of grid points M = Nx*Ny*Nz
    double *mask;                 ///< Mask array for impenetrable regions (0=blocked, 1=accessible)
    double *dv;                   ///< Volume element weights for integration (includes mask)
    double volume;                ///< Total system volume V = Lx*Ly*Lz
    double accessible_volume;     ///< Accessible volume (excluding masked regions)

    /**
     * @brief Boundary conditions for each face.
     *
     * Order: [x_low, x_high, y_low, y_high, z_low, z_high]
     * - For 1D: only indices 0,1 are used
     * - For 2D: indices 0-3 are used
     * - For 3D: all 6 indices are used
     */
    std::vector<BoundaryCondition> bc;

public:
    /**
     * @brief Construct a ComputationBox with specified grid and box parameters.
     *
     * @param nx   Number of grid points in each direction, e.g., {32, 32, 32}
     * @param lx   Box lengths in each direction, e.g., {4.0, 4.0, 4.0}
     * @param bc   Boundary condition strings: {"periodic", "reflecting", "absorbing"}
     *             Order: [x_low, x_high, y_low, y_high, z_low, z_high]
     * @param mask Optional mask array defining impenetrable regions.
     *             mask[i]=0 means grid point i is inside an obstacle.
     *             mask[i]=1 means grid point i is accessible.
     *             If nullptr, all points are accessible.
     *
     * @throws Exception if nx and lx have different sizes or invalid BC strings
     *
     * @example
     * @code
     * // 64x64x64 periodic box
     * std::vector<int> nx = {64, 64, 64};
     * std::vector<double> lx = {8.0, 8.0, 8.0};
     * std::vector<std::string> bc = {"periodic", "periodic", "periodic",
     *                                 "periodic", "periodic", "periodic"};
     * ComputationBox<double> cb(nx, lx, bc);
     * @endcode
     */
    ComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr);

    /**
     * @brief Virtual destructor.
     */
    virtual ~ComputationBox();

    /**
     * @brief Get spatial dimension.
     * @return Dimension (1, 2, or 3)
     */
    int get_dim();

    /**
     * @brief Get grid dimensions vector.
     * @return Vector of grid points [Nx, Ny, Nz]
     */
    std::vector<int> get_nx();

    /**
     * @brief Get grid dimension in one direction.
     * @param i Direction index (0=x, 1=y, 2=z)
     * @return Number of grid points in direction i
     */
    int get_nx(int i);

    /**
     * @brief Get box lengths vector.
     * @return Vector of box lengths [Lx, Ly, Lz] in units of a_Ref*N^0.5
     */
    std::vector<double> get_lx();

    /**
     * @brief Get box length in one direction.
     * @param i Direction index (0=x, 1=y, 2=z)
     * @return Box length in direction i
     */
    double get_lx(int i);

    /**
     * @brief Get grid spacing vector.
     * @return Vector of grid spacings [dx, dy, dz] = lx/nx
     */
    std::vector<double> get_dx();

    /**
     * @brief Get grid spacing in one direction.
     * @param i Direction index (0=x, 1=y, 2=z)
     * @return Grid spacing in direction i
     */
    double get_dx(int i);

    /**
     * @brief Get volume element weight at grid point.
     * @param i Grid point index
     * @return Volume element dV[i] (includes mask weighting)
     */
    double get_dv(int i);

    /**
     * @brief Get total number of grid points.
     * @return M = Nx*Ny*Nz
     */
    int get_total_grid();

    /**
     * @brief Get total system volume.
     * @return V = Lx*Ly*Lz in units of (a_Ref*N^0.5)^dim
     */
    double get_volume();

    /**
     * @brief Get accessible volume (excluding masked regions).
     * @return Accessible volume = sum of dV over unmasked points
     */
    double get_accessible_volume();

    /**
     * @brief Get pointer to mask array.
     * @return Const pointer to mask array (nullptr if no mask)
     */
    const double* get_mask() ;

    /**
     * @brief Get all boundary conditions.
     * @return Vector of BoundaryCondition enums
     */
    const std::vector<BoundaryCondition> get_boundary_conditions();

    /**
     * @brief Get boundary condition for specific face.
     * @param i Face index (0=x_low, 1=x_high, 2=y_low, ...)
     * @return BoundaryCondition enum value
     */
    BoundaryCondition get_boundary_condition(int i);

    /**
     * @brief Compute volume integral of a field.
     *
     * Computes: integral(g) = sum_i g[i] * dV[i]
     *
     * @param g Field array of length total_grid
     * @return Volume integral of g
     *
     * @example
     * @code
     * // Verify normalization: integral(phi) should equal volume fraction
     * double avg_phi = cb->integral(phi) / cb->get_volume();
     * @endcode
     */
    T integral(const T *g);

    /**
     * @brief Compute inner product of two fields.
     *
     * Computes: <g|h> = integral(g * h) = sum_i g[i] * h[i] * dV[i]
     *
     * @param g First field array of length total_grid
     * @param h Second field array of length total_grid
     * @return Inner product <g|h>
     *
     * @example
     * @code
     * // Compute Hamiltonian term: <w|phi>
     * double H_wp = cb->inner_product(w, phi);
     * @endcode
     */
    T inner_product(const T *g, const T *h);

    /**
     * @brief Compute weighted inner product with inverse weight.
     *
     * Computes: sum_i g[i] * h[i] / w[i] * dV[i]
     *
     * Useful for computing averages with non-uniform weights.
     *
     * @param g First field array
     * @param h Second field array
     * @param w Weight field array (must be non-zero everywhere)
     * @return Weighted inner product
     */
    T inner_product_inverse_weight(const T *g, const T *h, const T *w);

    /**
     * @brief Compute inner product for multiple field components.
     *
     * For multi-component fields g = [g1, g2, ...] and h = [h1, h2, ...],
     * computes: sum_i sum_c g[c*M+i] * h[c*M+i] * dV[i]
     *
     * @param n_comp Number of field components
     * @param g First multi-component field (length n_comp * total_grid)
     * @param h Second multi-component field (length n_comp * total_grid)
     * @return Sum of inner products over all components
     */
    T multi_inner_product(int n_comp, const T *g, const T *h);

    /**
     * @brief Remove the spatial mean from a field (make it zero-mean).
     *
     * Modifies g in-place: g[i] -= mean(g)
     *
     * @param g Field array (modified in-place)
     *
     * @example
     * @code
     * // Ensure pressure field has zero mean (gauge fixing)
     * cb->zero_mean(pressure);
     * @endcode
     */
    void zero_mean(T *g);

    /**
     * @brief Update box lengths (for box relaxation in SCFT).
     *
     * Updates lx, dx, dv, and volume. Used when optimizing box dimensions
     * to minimize free energy.
     *
     * @param new_lx New box lengths [Lx, Ly, Lz]
     *
     * @note After calling set_lx(), you should also update the Laplacian
     *       operator in the propagator solver.
     */
    virtual void set_lx(std::vector<double> new_lx);
};
#endif
