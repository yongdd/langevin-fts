/**
 * @file ComputationBox.cpp
 * @brief Implementation of ComputationBox simulation domain.
 *
 * Provides the abstract base implementation for managing the computational
 * grid, including grid dimensions, volume elements, boundary conditions,
 * and integral operations. Platform-specific FFT operations are implemented
 * in derived classes (CpuComputationBox, CudaComputationBox).
 *
 * **Grid Structure:**
 *
 * - Supports 1D, 2D, and 3D grids
 * - Grid spacing: dx[d] = lx[d] / nx[d]
 * - Volume element: dv[i] = Π dx[d] × mask[i]
 *
 * **Boundary Conditions:**
 *
 * Each dimension has two boundaries (low/high):
 * - PERIODIC: Wrap-around (default, required for pseudo-spectral)
 * - REFLECTING: Zero derivative (Neumann)
 * - ABSORBING: Zero value (Dirichlet)
 *
 * **Template Instantiations:**
 *
 * - ComputationBox<double>: Real-valued fields
 * - ComputationBox<std::complex<double>>: Complex-valued fields
 */

#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <map>
#include <complex>
#include <cmath>
#include <numbers>

#include "ComputationBox.h"
#include "SpaceGroup.h"
#include "ValidationUtils.h"

namespace {
    /// Tolerance for comparing mask values to 0.0 or 1.0
    constexpr double MASK_TOLERANCE = 1e-7;
    /// Tolerance for determining if lattice angles are orthogonal (90°)
    constexpr double ANGLE_TOLERANCE = 1e-10;
    /// Minimum sin(gamma) value to avoid division by zero in lattice calculations
    constexpr double SIN_GAMMA_MIN = 1e-10;

    /**
     * @brief Expand dimension-appropriate angles to full 3-element array.
     *
     * Accepts angles based on dimensionality:
     * - 1D: empty or {90} (no angles needed)
     * - 2D: {gamma} (only gamma matters, alpha=beta=90°)
     * - 3D: {alpha, beta, gamma}
     *
     * @param angles Input angles (0, 1, or 3 elements)
     * @param dim    Simulation dimensionality (1, 2, or 3)
     * @return Expanded 3-element angles array [alpha, beta, gamma]
     */
    std::vector<double> expand_angles(const std::vector<double>& angles, int dim)
    {
        std::vector<double> full_angles = {90.0, 90.0, 90.0};  // Default orthogonal

        if (angles.empty())
        {
            // No angles specified: orthogonal system
            return full_angles;
        }
        else if (angles.size() == 1)
        {
            // Single angle: gamma only (for 2D)
            if (dim != 2)
                throw_with_line_number("Single-angle specification is only valid for 2D systems")
            full_angles[2] = angles[0];  // gamma
            return full_angles;
        }
        else if (angles.size() == 3)
        {
            // Full specification
            return angles;
        }
        else
        {
            throw_with_line_number("angles must have 0, 1 (gamma for 2D), or 3 elements, but got " +
                std::to_string(angles.size()))
        }
    }
}

/**
 * @brief Construct computation box with given grid and boundaries.
 *
 * @param new_nx Grid points per dimension [nx, ny, nz]
 * @param new_lx Box lengths per dimension [Lx, Ly, Lz]
 * @param bc     Boundary conditions (2 per dimension: low, high)
 * @param mask   Optional mask for impenetrable regions (nullptr = no mask)
 *
 * @throws Exception if dimensions mismatch or invalid parameters
 */
template <typename T>
ComputationBox<T>::ComputationBox(std::vector<int> new_nx, std::vector<double> new_lx,
    std::vector<std::string> bc, const double* mask)
{
    validation::require_same_size(new_nx.size(), new_lx.size(), "nx", "lx");
    if ( new_nx.size() != 3 && new_nx.size() != 2 && new_nx.size() != 1)
    {
        throw_with_line_number("We expect 1D, 2D or 3D, but we get " + std::to_string(new_nx.size()))
    }
    validation::require_all_positive(new_nx, "nx");
    validation::require_all_positive(new_lx, "lx");

    try
    {
        this->dim = new_nx.size();
        nx = new_nx;
        lx = new_lx;

        const int DIM = this->dim;

        // Grid interval
        for(int d=0; d<DIM; d++)
            dx.push_back(lx[d]/nx[d]);

        // The number of grids
        total_grid = 1;
        for(int d=0; d<DIM; d++)
            total_grid *= nx[d];

        // Mask
        // Penetrable region == 1.0
        // Impenetrable region == 0.0
        if (mask != nullptr)
        {
            this->mask.assign(mask, mask + total_grid);
            validation::validate_mask(this->mask.data(), total_grid, MASK_TOLERANCE);
        }
        // else: mask remains empty (no mask)

        // Weight factor for integral
        dv.resize(total_grid);
        for(int i=0; i<total_grid; i++)
        {
            dv[i] = 1.0;
            for(int d=0; d<DIM; d++)
                dv[i] *= dx[d];
        }
        if (!this->mask.empty())
            for(int i=0; i<total_grid; i++)
                dv[i] *= this->mask[i];

        // Volume of simulation box
        volume = 1.0;
        for(int d=0; d<DIM; d++)
            volume *= lx[d];

        // Accessible volume
        accessible_volume = 0.0;
        for(int i=0; i<total_grid; i++)
            accessible_volume += dv[i];

        // Set boundary conditions
        if((unsigned int) 2*DIM != bc.size() && 0 != bc.size())
        {
            throw_with_line_number(
                "We expect 0 or " + std::to_string(2*DIM) + " boundary conditions, but we get " + std::to_string(bc.size()) +
                ". For each dimension, two boundary conditions (top and bottom) are required.");
        }

        if(bc.size() == 0)
        {
            // Default is periodic boundary
            for(int i=0; i<2*DIM; i++)
                this->bc.push_back(BoundaryCondition::PERIODIC);
        }
        else
        {
            for(int i=0; i<2*DIM; i+=2)
            {
                std::string bc_name_l = validation::to_lower(bc[i]);
                std::string bc_name_h = validation::to_lower(bc[i+1]);

                if((bc_name_l == "periodic" && bc_name_h != "periodic") ||
                   (bc_name_l != "periodic" && bc_name_h == "periodic"))
                {
                   throw_with_line_number(bc_name_l + " and "  + bc_name_h + " are an invalid boundary condition combination. " +
                    + "If one side imposes a periodic boundary condition, the other side must also be a periodic boundary condition.")
                }
            }

            for(int i=0; i<2*DIM; i++)
            {
                std::string bc_name = validation::to_lower(bc[i]);

                if(bc_name == "periodic")
                    this->bc.push_back(BoundaryCondition::PERIODIC);
                else if(bc_name == "reflecting")
                    this->bc.push_back(BoundaryCondition::REFLECTING);
                else if(bc_name == "absorbing")
                    this->bc.push_back(BoundaryCondition::ABSORBING);
                else
                {
                    throw_with_line_number(bc_name + " is an invalid boundary condition. Choose among ['periodic', 'reflecting', 'absorbing']")
                }
            }
        }

        // Initialize default orthogonal angles (90 degrees = π/2 radians)
        angles_.resize(3);
        for(int d=0; d<3; d++)
            angles_[d] = std::numbers::pi / 2.0;
        is_orthogonal_ = true;

        // Initialize lattice vectors and metric tensor
        lattice_vec_.fill(0.0);
        recip_vec_.fill(0.0);
        recip_metric_.fill(0.0);

        // Compute lattice vectors, reciprocal lattice, and metric tensor
        compute_lattice_vectors();
        compute_reciprocal_lattice();
        compute_recip_metric();

        // Initialize space group (disabled by default)
        space_group_ = nullptr;
        n_irreducible_ = total_grid;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
//----------------- Constructor with angles -----------------------------
/**
 * @brief Construct computation box with lattice angles for non-orthogonal systems.
 *
 * @param new_nx    Grid points per dimension [nx, ny, nz]
 * @param new_lx    Box lengths (lattice constants) per dimension [a, b, c]
 * @param bc        Boundary conditions (2 per dimension: low, high)
 * @param angles    Lattice angles [alpha, beta, gamma] in DEGREES
 * @param mask      Optional mask for impenetrable regions (nullptr = no mask)
 *
 * @throws Exception if dimensions mismatch, invalid parameters, or invalid angles
 */
template <typename T>
ComputationBox<T>::ComputationBox(std::vector<int> new_nx, std::vector<double> new_lx,
    std::vector<std::string> bc, std::vector<double> angles, const double* mask)
{
    validation::require_same_size(new_nx.size(), new_lx.size(), "nx", "lx");
    if ( new_nx.size() != 3 && new_nx.size() != 2 && new_nx.size() != 1)
    {
        throw_with_line_number("We expect 1D, 2D or 3D, but we get " + std::to_string(new_nx.size()))
    }
    validation::require_all_positive(new_nx, "nx");
    validation::require_all_positive(new_lx, "lx");

    try
    {
        this->dim = new_nx.size();
        nx = new_nx;
        lx = new_lx;

        const int DIM = this->dim;

        // Expand angles to full 3-element array (handles 1-element gamma for 2D)
        std::vector<double> full_angles = expand_angles(angles, DIM);

        // Validate angles
        for (int i = 0; i < 3; i++) {
            validation::require_in_range(full_angles[i], 0.0 + 1e-10, 180.0 - 1e-10,
                "angles[" + std::to_string(i) + "]");
        }

        // Convert angles from degrees to radians and store
        angles_.resize(3);
        const double deg_to_rad = std::numbers::pi / 180.0;
        for (int d = 0; d < 3; d++)
            angles_[d] = full_angles[d] * deg_to_rad;

        // Check if orthogonal (all angles are 90 degrees within tolerance)
        const double angle_tolerance = ANGLE_TOLERANCE;
        is_orthogonal_ = (std::abs(angles_[0] - std::numbers::pi/2.0) < angle_tolerance &&
                          std::abs(angles_[1] - std::numbers::pi/2.0) < angle_tolerance &&
                          std::abs(angles_[2] - std::numbers::pi/2.0) < angle_tolerance);

        // Validate that angles can form a valid unit cell
        // For a valid cell: the volume must be positive
        // V = abc * sqrt(1 - cos²α - cos²β - cos²γ + 2*cosα*cosβ*cosγ)
        double cos_a = std::cos(angles_[0]);
        double cos_b = std::cos(angles_[1]);
        double cos_g = std::cos(angles_[2]);
        double vol_factor = 1.0 - cos_a*cos_a - cos_b*cos_b - cos_g*cos_g + 2.0*cos_a*cos_b*cos_g;
        if (vol_factor <= 0.0)
            throw_with_line_number("Invalid lattice angles: they do not form a valid unit cell. "
                                   "The combination (alpha=" + std::to_string(angles[0]) +
                                   ", beta=" + std::to_string(angles[1]) +
                                   ", gamma=" + std::to_string(angles[2]) + ") degrees is geometrically impossible.");

        // Grid interval
        for(int d=0; d<DIM; d++)
            dx.push_back(lx[d]/nx[d]);

        // The number of grids
        total_grid = 1;
        for(int d=0; d<DIM; d++)
            total_grid *= nx[d];

        // Mask
        if (mask != nullptr)
        {
            this->mask.assign(mask, mask + total_grid);
            validation::validate_mask(this->mask.data(), total_grid, MASK_TOLERANCE);
        }
        // else: mask remains empty (no mask)

        // Initialize lattice vectors and metric tensor
        lattice_vec_.fill(0.0);
        recip_vec_.fill(0.0);
        recip_metric_.fill(0.0);

        // Compute lattice vectors, reciprocal lattice, and metric tensor
        compute_lattice_vectors();
        compute_reciprocal_lattice();
        compute_recip_metric();

        // Weight factor for integral (using volume from lattice vectors)
        dv.resize(total_grid);
        double dv_base = volume / total_grid;  // Volume element
        for(int i=0; i<total_grid; i++)
            dv[i] = dv_base;
        if (!this->mask.empty())
            for(int i=0; i<total_grid; i++)
                dv[i] *= this->mask[i];

        // Accessible volume
        accessible_volume = 0.0;
        for(int i=0; i<total_grid; i++)
            accessible_volume += dv[i];

        // Set boundary conditions
        if((unsigned int) 2*DIM != bc.size() && 0 != bc.size())
        {
            throw_with_line_number(
                "We expect 0 or " + std::to_string(2*DIM) + " boundary conditions, but we get " + std::to_string(bc.size()) +
                ". For each dimension, two boundary conditions (top and bottom) are required.");
        }

        if(bc.size() == 0)
        {
            for(int i=0; i<2*DIM; i++)
                this->bc.push_back(BoundaryCondition::PERIODIC);
        }
        else
        {
            for(int i=0; i<2*DIM; i+=2)
            {
                std::string bc_name_l = validation::to_lower(bc[i]);
                std::string bc_name_h = validation::to_lower(bc[i+1]);

                if((bc_name_l == "periodic" && bc_name_h != "periodic") ||
                   (bc_name_l != "periodic" && bc_name_h == "periodic"))
                {
                   throw_with_line_number(bc_name_l + " and "  + bc_name_h + " are an invalid boundary condition combination. " +
                    + "If one side imposes a periodic boundary condition, the other side must also be a periodic boundary condition.")
                }
            }

            for(int i=0; i<2*DIM; i++)
            {
                std::string bc_name = validation::to_lower(bc[i]);

                if(bc_name == "periodic")
                    this->bc.push_back(BoundaryCondition::PERIODIC);
                else if(bc_name == "reflecting")
                    this->bc.push_back(BoundaryCondition::REFLECTING);
                else if(bc_name == "absorbing")
                    this->bc.push_back(BoundaryCondition::ABSORBING);
                else
                {
                    throw_with_line_number(bc_name + " is an invalid boundary condition. Choose among ['periodic', 'reflecting', 'absorbing']")
                }
            }
        }

        // Initialize space group (disabled by default)
        space_group_ = nullptr;
        n_irreducible_ = total_grid;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//----------------- Destructor -----------------------------
template <typename T>
ComputationBox<T>::~ComputationBox()
{
    // Vectors handle memory cleanup automatically
}

//----------------- Lattice computation methods -----------------------------
/**
 * @brief Compute lattice vectors from box lengths and angles.
 *
 * Standard crystallographic parameterization:
 * - a along x-axis: a = (a, 0, 0)
 * - b in xy-plane:  b = (b*cos(γ), b*sin(γ), 0)
 * - c general:      c = (c_x, c_y, c_z)
 *
 * where c_x = c*cos(β), c_y = c*(cos(α) - cos(β)*cos(γ))/sin(γ),
 * and c_z = V / (a * b * sin(γ))
 *
 * Also computes the cell volume: V = a·(b×c)
 */
template <typename T>
void ComputationBox<T>::compute_lattice_vectors()
{
    // For 1D and 2D, we still use the 3D formalism but with unit values
    double a = (dim >= 1) ? lx[0] : 1.0;
    double b = (dim >= 2) ? lx[1] : 1.0;
    double c = (dim >= 3) ? lx[2] : 1.0;

    double alpha = angles_[0];  // angle between b and c
    double beta = angles_[1];   // angle between a and c
    double gamma = angles_[2];  // angle between a and b

    double cos_a = std::cos(alpha);
    double cos_b = std::cos(beta);
    double cos_g = std::cos(gamma);
    double sin_g = std::sin(gamma);

    // Compute volume factor: V = abc * sqrt(1 - cos²α - cos²β - cos²γ + 2*cosα*cosβ*cosγ)
    double vol_factor_sq = 1.0 - cos_a*cos_a - cos_b*cos_b - cos_g*cos_g + 2.0*cos_a*cos_b*cos_g;
    double vol_factor = std::sqrt(vol_factor_sq);

    // Cell volume
    volume = a * b * c * vol_factor;

    // Lattice vector a (along x-axis)
    lattice_vec_[0] = a;
    lattice_vec_[1] = 0.0;
    lattice_vec_[2] = 0.0;

    // Lattice vector b (in xy-plane)
    lattice_vec_[3] = b * cos_g;
    lattice_vec_[4] = b * sin_g;
    lattice_vec_[5] = 0.0;

    // Lattice vector c
    double c_x = c * cos_b;
    double c_y = (sin_g > SIN_GAMMA_MIN) ? c * (cos_a - cos_b * cos_g) / sin_g : 0.0;
    double c_z = volume / (a * b * sin_g);

    lattice_vec_[6] = c_x;
    lattice_vec_[7] = c_y;
    lattice_vec_[8] = c_z;
}

/**
 * @brief Compute reciprocal lattice vectors from direct lattice.
 *
 * Reciprocal vectors are defined as:
 * - a* = (b × c) / V
 * - b* = (c × a) / V
 * - c* = (a × b) / V
 *
 * where V = a·(b×c) is the cell volume.
 *
 * The reciprocal vectors satisfy: a*·a = b*·b = c*·c = 1
 * and a*·b = a*·c = b*·a = b*·c = c*·a = c*·b = 0
 *
 * Note: These are the "crystallographer's" reciprocal vectors (without 2π factor).
 */
template <typename T>
void ComputationBox<T>::compute_reciprocal_lattice()
{
    // Direct lattice vectors
    double ax = lattice_vec_[0], ay = lattice_vec_[1], az = lattice_vec_[2];
    double bx = lattice_vec_[3], by = lattice_vec_[4], bz = lattice_vec_[5];
    double cx = lattice_vec_[6], cy = lattice_vec_[7], cz = lattice_vec_[8];

    // b × c
    double bc_x = by * cz - bz * cy;
    double bc_y = bz * cx - bx * cz;
    double bc_z = bx * cy - by * cx;

    // c × a
    double ca_x = cy * az - cz * ay;
    double ca_y = cz * ax - cx * az;
    double ca_z = cx * ay - cy * ax;

    // a × b
    double ab_x = ay * bz - az * by;
    double ab_y = az * bx - ax * bz;
    double ab_z = ax * by - ay * bx;

    // a* = (b × c) / V
    recip_vec_[0] = bc_x / volume;
    recip_vec_[1] = bc_y / volume;
    recip_vec_[2] = bc_z / volume;

    // b* = (c × a) / V
    recip_vec_[3] = ca_x / volume;
    recip_vec_[4] = ca_y / volume;
    recip_vec_[5] = ca_z / volume;

    // c* = (a × b) / V
    recip_vec_[6] = ab_x / volume;
    recip_vec_[7] = ab_y / volume;
    recip_vec_[8] = ab_z / volume;
}

/**
 * @brief Compute reciprocal metric tensor from reciprocal vectors.
 *
 * The reciprocal metric tensor G*_ij = e*_i · e*_j where e* are reciprocal basis vectors.
 *
 * For wavenumber magnitude: |k|² = G*_ij k_i k_j
 *                                = G*_00 kx² + G*_11 ky² + G*_22 kz²
 *                                  + 2*G*_01 kx*ky + 2*G*_02 kx*kz + 2*G*_12 ky*kz
 *
 * Storage layout: [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22]
 */
template <typename T>
void ComputationBox<T>::compute_recip_metric()
{
    // Reciprocal lattice vectors
    double a_star_x = recip_vec_[0], a_star_y = recip_vec_[1], a_star_z = recip_vec_[2];
    double b_star_x = recip_vec_[3], b_star_y = recip_vec_[4], b_star_z = recip_vec_[5];
    double c_star_x = recip_vec_[6], c_star_y = recip_vec_[7], c_star_z = recip_vec_[8];

    // G*_00 = a* · a*
    recip_metric_[0] = a_star_x * a_star_x + a_star_y * a_star_y + a_star_z * a_star_z;

    // G*_01 = a* · b*
    recip_metric_[1] = a_star_x * b_star_x + a_star_y * b_star_y + a_star_z * b_star_z;

    // G*_02 = a* · c*
    recip_metric_[2] = a_star_x * c_star_x + a_star_y * c_star_y + a_star_z * c_star_z;

    // G*_11 = b* · b*
    recip_metric_[3] = b_star_x * b_star_x + b_star_y * b_star_y + b_star_z * b_star_z;

    // G*_12 = b* · c*
    recip_metric_[4] = b_star_x * c_star_x + b_star_y * c_star_y + b_star_z * c_star_z;

    // G*_22 = c* · c*
    recip_metric_[5] = c_star_x * c_star_x + c_star_y * c_star_y + c_star_z * c_star_z;
}

//----------------- get methods-------------------------------------
template <typename T>
int ComputationBox<T>::get_dim() const
{
    return dim;
}
template <typename T>
int ComputationBox<T>::get_nx(int i) const
{
    if (i < 0 or i >= dim)
    {
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.")
    }
    return nx[i];
}
template <typename T>
double ComputationBox<T>::get_lx(int i) const
{
    if (i < 0 or i >= dim)
    {
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.")
    }
    return lx[i];
}
template <typename T>
double ComputationBox<T>::get_dx(int i) const
{
    if (i < 0 or i >= dim)
    {
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.")
    }
    return dx[i];
}
template <typename T>
std::vector<int> ComputationBox<T>::get_nx() const
{
    return nx;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_lx() const
{
    return lx;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_dx() const
{
    return dx;
}
template <typename T>
double ComputationBox<T>::get_dv(int i) const
{
    return dv[i];
}
template <typename T>
int ComputationBox<T>::get_total_grid() const
{
    return total_grid;
}
template <typename T>
double ComputationBox<T>::get_volume() const
{
    return volume;
}
template <typename T>
double ComputationBox<T>::get_accessible_volume() const
{
    return accessible_volume;
}
template <typename T>
const double* ComputationBox<T>::get_mask() const
{
    return mask.empty() ? nullptr : mask.data();
}
template <typename T>
const std::vector<BoundaryCondition> ComputationBox<T>::get_boundary_conditions() const
{
    return bc;
}
template <typename T>
BoundaryCondition ComputationBox<T>::get_boundary_condition(int i) const
{
    return bc[i];
}
template <typename T>
std::vector<double> ComputationBox<T>::get_angles() const
{
    return angles_;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_angles_degrees() const
{
    const double rad_to_deg = 180.0 / std::numbers::pi;
    std::vector<double> angles_deg(3);
    for (int d = 0; d < 3; d++)
        angles_deg[d] = angles_[d] * rad_to_deg;
    return angles_deg;
}
template <typename T>
const std::array<double, 9>& ComputationBox<T>::get_lattice_vec() const
{
    return lattice_vec_;
}
template <typename T>
const std::array<double, 9>& ComputationBox<T>::get_recip_vec() const
{
    return recip_vec_;
}
template <typename T>
const std::array<double, 6>& ComputationBox<T>::get_recip_metric() const
{
    return recip_metric_;
}
template <typename T>
bool ComputationBox<T>::is_orthogonal() const
{
    return is_orthogonal_;
}
//----------------- set methods-------------------------------------
template <typename T>
void ComputationBox<T>::set_lx(std::vector<double> new_lx)
{
    validation::require_same_size(new_lx.size(), (size_t)dim, "new lx", "dim");
    validation::require_all_positive(new_lx, "new lx");

    lx = new_lx;

    // Grid interval
    for(int d=0; d<dim; d++)
        dx[d] = lx[d]/nx[d];

    // Recompute lattice vectors, reciprocal lattice, and metric tensor
    compute_lattice_vectors();
    compute_reciprocal_lattice();
    compute_recip_metric();

    // Weight factor for integral (using volume from lattice vectors)
    double dv_base = volume / total_grid;
    for(int i=0; i<total_grid; i++)
        dv[i] = dv_base;
    if (!this->mask.empty())
        for(int i=0; i<total_grid; i++)
            dv[i] *= this->mask[i];
}

template <typename T>
void ComputationBox<T>::set_lattice_parameters(std::vector<double> new_lx, std::vector<double> new_angles)
{
    validation::require_same_size(new_lx.size(), (size_t)dim, "new lx", "dim");
    validation::require_all_positive(new_lx, "new lx");

    // Expand angles to full 3-element array (handles 1-element gamma for 2D)
    std::vector<double> full_angles = expand_angles(new_angles, dim);

    // Validate angles
    for (int i = 0; i < 3; i++) {
        validation::require_in_range(full_angles[i], 0.0 + 1e-10, 180.0 - 1e-10,
            "angles[" + std::to_string(i) + "]");
    }

    lx = new_lx;

    // Convert angles from degrees to radians and store
    const double deg_to_rad = std::numbers::pi / 180.0;
    for (int d = 0; d < 3; d++)
        angles_[d] = full_angles[d] * deg_to_rad;

    // Check if orthogonal
    const double angle_tolerance = ANGLE_TOLERANCE;
    is_orthogonal_ = (std::abs(angles_[0] - std::numbers::pi/2.0) < angle_tolerance &&
                      std::abs(angles_[1] - std::numbers::pi/2.0) < angle_tolerance &&
                      std::abs(angles_[2] - std::numbers::pi/2.0) < angle_tolerance);

    // Validate that angles can form a valid unit cell
    double cos_a = std::cos(angles_[0]);
    double cos_b = std::cos(angles_[1]);
    double cos_g = std::cos(angles_[2]);
    double vol_factor = 1.0 - cos_a*cos_a - cos_b*cos_b - cos_g*cos_g + 2.0*cos_a*cos_b*cos_g;
    if (vol_factor <= 0.0)
        throw_with_line_number("Invalid lattice angles: they do not form a valid unit cell.");

    // Grid interval
    for(int d=0; d<dim; d++)
        dx[d] = lx[d]/nx[d];

    // Recompute lattice vectors, reciprocal lattice, and metric tensor
    compute_lattice_vectors();
    compute_reciprocal_lattice();
    compute_recip_metric();

    // Weight factor for integral (using volume from lattice vectors)
    double dv_base = volume / total_grid;
    for(int i=0; i<total_grid; i++)
        dv[i] = dv_base;
    if (!this->mask.empty())
        for(int i=0; i<total_grid; i++)
            dv[i] *= this->mask[i];
}
// ==================== Space Group Methods ====================

template <typename T>
void ComputationBox<T>::set_space_group(SpaceGroup* sg)
{
    space_group_ = sg;

    if (sg != nullptr)
    {
        // Cache orbit counts and n_irreducible for fast access
        orbit_counts_ = sg->get_orbit_counts();
        n_irreducible_ = sg->get_n_irreducible();

        // Convert mask to reduced basis
        if (!mask.empty())
        {
            std::vector<double> mask_reduced(n_irreducible_);
            sg->to_reduced_basis(mask.data(), mask_reduced.data(), 1);
            mask = std::move(mask_reduced);
        }
    }
    else
    {
        // Clear cached values
        orbit_counts_.clear();
        n_irreducible_ = total_grid;
    }
}

template <typename T>
SpaceGroup* ComputationBox<T>::get_space_group() const
{
    return space_group_;
}

template <typename T>
int ComputationBox<T>::get_n_basis() const
{
    return n_irreducible_;
}

template <typename T>
const std::vector<int>& ComputationBox<T>::get_orbit_counts() const
{
    return orbit_counts_;
}

// ==================== Field Operations ====================

/**
 * @brief Compute volume integral of a field.
 *
 * When space_group_ is set, operates on reduced basis (size n_irreducible).
 * Otherwise, operates on full grid (size total_grid).
 */
template <typename T>
T ComputationBox<T>::integral(const T *g)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode
        double dv_base = volume / total_grid;
        for (int i = 0; i < n_irreducible_; i++)
            sum += static_cast<T>(orbit_counts_[i]) * g[i];
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        // Full grid mode
        for (int i = 0; i < total_grid; i++)
            sum += dv[i] * g[i];
        return sum;
    }
}

/**
 * @brief Compute mean (spatial average) of a field.
 */
template <typename T>
T ComputationBox<T>::mean(const T *g)
{
    return integral(g) / static_cast<T>(volume);
}

/**
 * @brief Compute inner product of two fields.
 *
 * When space_group_ is set, operates on reduced basis.
 */
template <typename T>
T ComputationBox<T>::inner_product(const T *g, const T *h)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode
        double dv_base = volume / total_grid;
        for (int i = 0; i < n_irreducible_; i++)
            sum += static_cast<T>(orbit_counts_[i]) * g[i] * h[i];
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        // Full grid mode
        for (int i = 0; i < total_grid; i++)
            sum += dv[i] * g[i] * h[i];
        return sum;
    }
}

/**
 * @brief Compute weighted inner product with inverse weight.
 *
 * When space_group_ is set, operates on reduced basis.
 */
template <typename T>
T ComputationBox<T>::inner_product_inverse_weight(const T *g, const T *h, const T *w)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode
        double dv_base = volume / total_grid;
        for (int i = 0; i < n_irreducible_; i++)
            sum += static_cast<T>(orbit_counts_[i]) * g[i] * h[i] / w[i];
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        // Full grid mode
        for (int i = 0; i < total_grid; i++)
            sum += dv[i] * g[i] * h[i] / w[i];
        return sum;
    }
}

/**
 * @brief Compute inner product for multiple field components.
 *
 * When space_group_ is set, operates on reduced basis.
 */
template <typename T>
T ComputationBox<T>::multi_inner_product(int n_comp, const T *g, const T *h)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode
        double dv_base = volume / total_grid;
        for (int n = 0; n < n_comp; n++)
        {
            for (int i = 0; i < n_irreducible_; i++)
                sum += static_cast<T>(orbit_counts_[i]) * g[i + n * n_irreducible_] * h[i + n * n_irreducible_];
        }
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        // Full grid mode
        for (int n = 0; n < n_comp; n++)
        {
            for (int i = 0; i < total_grid; i++)
                sum += dv[i] * g[i + n * total_grid] * h[i + n * total_grid];
        }
        return sum;
    }
}

/**
 * @brief Remove the spatial mean from a field (make it zero-mean).
 *
 * When space_group_ is set, operates on reduced basis using weighted mean.
 */
template <typename T>
void ComputationBox<T>::zero_mean(T *g)
{
    T sum{0.0};

    if (space_group_ != nullptr)
    {
        // Reduced basis mode: compute weighted mean
        int total_count = 0;
        for (int i = 0; i < n_irreducible_; i++)
        {
            sum += static_cast<T>(orbit_counts_[i]) * g[i];
            total_count += orbit_counts_[i];
        }
        T mean = sum / static_cast<T>(total_count);

        for (int i = 0; i < n_irreducible_; i++)
            g[i] -= mean;
    }
    else
    {
        // Full grid mode
        for (int i = 0; i < total_grid; i++)
            sum += dv[i] * g[i];
        for (int i = 0; i < total_grid; i++)
            g[i] -= sum / volume;
    }
}

// Explicit template instantiation
template class ComputationBox<double>;
template class ComputationBox<std::complex<double>>;