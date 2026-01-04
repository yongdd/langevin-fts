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
    if ( new_nx.size() != new_lx.size() )
        throw_with_line_number("The sizes of nx (" + std::to_string(new_nx.size()) + ") and lx (" + std::to_string(new_lx.size()) + ") must match.");
    if ( new_nx.size() != 3 && new_nx.size() != 2 && new_nx.size() != 1)
        throw_with_line_number("We expect 1D, 2D or 3D, but we get " + std::to_string(new_nx.size()));
    if (std::any_of(new_nx.begin(), new_nx.end(), [](int nx) { return nx <= 0;})){
        std::stringstream ss_nx;
        std::copy(new_nx.begin(), new_nx.end(), std::ostream_iterator<int>(ss_nx, ", "));
        std::string str_nx = ss_nx.str();
        str_nx = str_nx.substr(0, str_nx.length()-2);
        throw_with_line_number("nx (" + str_nx + ") must be positive numbers");
    }
    if (std::any_of(new_lx.begin(), new_lx.end(), [](double lx) { return lx <= 0.0;})){
        std::stringstream ss_lx;
        std::copy(new_lx.begin(), new_lx.end(), std::ostream_iterator<int>(ss_lx, ", "));
        std::string str_lx = ss_lx.str();
        str_lx = str_lx.substr(0, str_lx.length()-2);
        throw_with_line_number("lx (" + str_lx + ") must be positive numbers");
    }

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
            this->mask = new double[total_grid];
            for(int i=0; i<total_grid; i++)
            {
                if(std::abs(mask[i]) < 1e-7)
                    this->mask[i] = 0.0;
                else if(std::abs(mask[i]-1.0) < 1e-7)
                    this->mask[i] = 1.0;
                else
                    throw_with_line_number("mask[" + std::to_string(i) + "] must be 0.0 or 1.0");
            }
        }
        else
            this->mask = nullptr;

        // Weight factor for integral
        dv = new double[total_grid];
        for(int i=0; i<total_grid; i++)
        {
            dv[i] = 1.0;
            for(int d=0; d<DIM; d++)
                dv[i] *= dx[d];
        }
        if (this->mask != nullptr)
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
                std::string bc_name_l = bc[i];
                std::string bc_name_h = bc[i+1];
                // Transform into lower cases
                std::transform(bc_name_l.begin(), bc_name_l.end(), bc_name_l.begin(),
                            [](unsigned char c)
                {
                    return std::tolower(c);
                });

                std::transform(bc_name_h.begin(), bc_name_h.end(), bc_name_h.begin(),
                            [](unsigned char c)
                {
                    return std::tolower(c);
                });

                if((bc_name_l == "periodic" && bc_name_h != "periodic") ||
                   (bc_name_l != "periodic" && bc_name_h == "periodic"))
                {
                   throw_with_line_number(bc_name_l + " and "  + bc_name_h + " are an invalid boundary condition combination. " +
                    + "If one side imposes a periodic boundary condition, the other side must also be a periodic boundary condition.");
                }

            }

            for(int i=0; i<2*DIM; i++)
            {
                std::string bc_name = bc[i];
                // Transform into lower cases
                std::transform(bc_name.begin(), bc_name.end(), bc_name.begin(),
                            [](unsigned char c)
                {
                    return std::tolower(c);
                });

                if(bc_name == "periodic")
                    this->bc.push_back(BoundaryCondition::PERIODIC);
                else if(bc_name == "reflecting")
                    this->bc.push_back(BoundaryCondition::REFLECTING);
                else if(bc_name == "absorbing")
                    this->bc.push_back(BoundaryCondition::ABSORBING);
                else
                    throw_with_line_number(bc_name + " is an invalid boundary condition. Choose among ['periodic', 'reflecting', 'absorbing']");
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
    if ( new_nx.size() != new_lx.size() )
        throw_with_line_number("The sizes of nx (" + std::to_string(new_nx.size()) + ") and lx (" + std::to_string(new_lx.size()) + ") must match.");
    if ( new_nx.size() != 3 && new_nx.size() != 2 && new_nx.size() != 1)
        throw_with_line_number("We expect 1D, 2D or 3D, but we get " + std::to_string(new_nx.size()));
    if (std::any_of(new_nx.begin(), new_nx.end(), [](int nx) { return nx <= 0;})){
        std::stringstream ss_nx;
        std::copy(new_nx.begin(), new_nx.end(), std::ostream_iterator<int>(ss_nx, ", "));
        std::string str_nx = ss_nx.str();
        str_nx = str_nx.substr(0, str_nx.length()-2);
        throw_with_line_number("nx (" + str_nx + ") must be positive numbers");
    }
    if (std::any_of(new_lx.begin(), new_lx.end(), [](double lx) { return lx <= 0.0;})){
        std::stringstream ss_lx;
        std::copy(new_lx.begin(), new_lx.end(), std::ostream_iterator<int>(ss_lx, ", "));
        std::string str_lx = ss_lx.str();
        str_lx = str_lx.substr(0, str_lx.length()-2);
        throw_with_line_number("lx (" + str_lx + ") must be positive numbers");
    }

    // Validate angles
    if (angles.size() != 3)
        throw_with_line_number("angles must have exactly 3 elements [alpha, beta, gamma], but got " + std::to_string(angles.size()));

    for (int i = 0; i < 3; i++) {
        if (angles[i] <= 0.0 || angles[i] >= 180.0)
            throw_with_line_number("angles[" + std::to_string(i) + "] = " + std::to_string(angles[i]) +
                                   " is invalid. Angles must be in range (0, 180) degrees.");
    }

    try
    {
        this->dim = new_nx.size();
        nx = new_nx;
        lx = new_lx;

        const int DIM = this->dim;

        // Convert angles from degrees to radians and store
        angles_.resize(3);
        const double deg_to_rad = std::numbers::pi / 180.0;
        for (int d = 0; d < 3; d++)
            angles_[d] = angles[d] * deg_to_rad;

        // Check if orthogonal (all angles are 90 degrees within tolerance)
        const double angle_tolerance = 1e-10;
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
            this->mask = new double[total_grid];
            for(int i=0; i<total_grid; i++)
            {
                if(std::abs(mask[i]) < 1e-7)
                    this->mask[i] = 0.0;
                else if(std::abs(mask[i]-1.0) < 1e-7)
                    this->mask[i] = 1.0;
                else
                    throw_with_line_number("mask[" + std::to_string(i) + "] must be 0.0 or 1.0");
            }
        }
        else
            this->mask = nullptr;

        // Initialize lattice vectors and metric tensor
        lattice_vec_.fill(0.0);
        recip_vec_.fill(0.0);
        recip_metric_.fill(0.0);

        // Compute lattice vectors, reciprocal lattice, and metric tensor
        compute_lattice_vectors();
        compute_reciprocal_lattice();
        compute_recip_metric();

        // Weight factor for integral (using volume from lattice vectors)
        dv = new double[total_grid];
        double dv_base = volume / total_grid;  // Volume element
        for(int i=0; i<total_grid; i++)
            dv[i] = dv_base;
        if (this->mask != nullptr)
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
                std::string bc_name_l = bc[i];
                std::string bc_name_h = bc[i+1];
                std::transform(bc_name_l.begin(), bc_name_l.end(), bc_name_l.begin(),
                            [](unsigned char c) { return std::tolower(c); });
                std::transform(bc_name_h.begin(), bc_name_h.end(), bc_name_h.begin(),
                            [](unsigned char c) { return std::tolower(c); });

                if((bc_name_l == "periodic" && bc_name_h != "periodic") ||
                   (bc_name_l != "periodic" && bc_name_h == "periodic"))
                {
                   throw_with_line_number(bc_name_l + " and "  + bc_name_h + " are an invalid boundary condition combination. " +
                    + "If one side imposes a periodic boundary condition, the other side must also be a periodic boundary condition.");
                }
            }

            for(int i=0; i<2*DIM; i++)
            {
                std::string bc_name = bc[i];
                std::transform(bc_name.begin(), bc_name.end(), bc_name.begin(),
                            [](unsigned char c) { return std::tolower(c); });

                if(bc_name == "periodic")
                    this->bc.push_back(BoundaryCondition::PERIODIC);
                else if(bc_name == "reflecting")
                    this->bc.push_back(BoundaryCondition::REFLECTING);
                else if(bc_name == "absorbing")
                    this->bc.push_back(BoundaryCondition::ABSORBING);
                else
                    throw_with_line_number(bc_name + " is an invalid boundary condition. Choose among ['periodic', 'reflecting', 'absorbing']");
            }
        }
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
    delete[] dv;
    if (mask != nullptr)
        delete[] mask;
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
    double c_y = (sin_g > 1e-10) ? c * (cos_a - cos_b * cos_g) / sin_g : 0.0;
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
int ComputationBox<T>::get_dim()
{
    return dim;
}
template <typename T>
int ComputationBox<T>::get_nx(int i)
{
    if (i < 0 or i >= dim)
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.");
    return nx[i];
}
template <typename T>
double ComputationBox<T>::get_lx(int i)
{
    if (i < 0 or i >= dim)
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.");
    return lx[i];
}
template <typename T>
double ComputationBox<T>::get_dx(int i)
{
    if (i < 0 or i >= dim)
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.");
    return dx[i];
}
template <typename T>
std::vector<int> ComputationBox<T>::get_nx()
{
    return nx;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_lx()
{
    return lx;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_dx()
{
    return dx;
}
template <typename T>
double ComputationBox<T>::get_dv(int i)
{
    return dv[i];
}
template <typename T>
int ComputationBox<T>::get_total_grid()
{
    return total_grid;
}
template <typename T>
double ComputationBox<T>::get_volume()
{
    return volume;
}
template <typename T>
double ComputationBox<T>::get_accessible_volume()
{
    return accessible_volume;
}
template <typename T>
const double* ComputationBox<T>::get_mask()
{
    return mask;
}
template <typename T>
const std::vector<BoundaryCondition> ComputationBox<T>::get_boundary_conditions()
{
    return bc;
}
template <typename T>
BoundaryCondition ComputationBox<T>::get_boundary_condition(int i)
{
    return bc[i];
}
template <typename T>
std::vector<double> ComputationBox<T>::get_angles()
{
    return angles_;
}
template <typename T>
std::vector<double> ComputationBox<T>::get_angles_degrees()
{
    const double rad_to_deg = 180.0 / std::numbers::pi;
    std::vector<double> angles_deg(3);
    for (int d = 0; d < 3; d++)
        angles_deg[d] = angles_[d] * rad_to_deg;
    return angles_deg;
}
template <typename T>
const std::array<double, 9>& ComputationBox<T>::get_lattice_vec()
{
    return lattice_vec_;
}
template <typename T>
const std::array<double, 9>& ComputationBox<T>::get_recip_vec()
{
    return recip_vec_;
}
template <typename T>
const std::array<double, 6>& ComputationBox<T>::get_recip_metric()
{
    return recip_metric_;
}
template <typename T>
bool ComputationBox<T>::is_orthogonal()
{
    return is_orthogonal_;
}
//----------------- set methods-------------------------------------
template <typename T>
void ComputationBox<T>::set_lx(std::vector<double> new_lx)
{
    if ( new_lx.size() != (unsigned int) dim )
        throw_with_line_number("The sizes of new lx (" + std::to_string(new_lx.size()) + ") and dim (" + std::to_string(dim) + ") must match.");

    if (std::any_of(new_lx.begin(), new_lx.end(), [](double lx) { return lx <= 0.0;})){
        std::stringstream ss_lx;
        std::copy(new_lx.begin(), new_lx.end(), std::ostream_iterator<int>(ss_lx, ", "));
        std::string str_lx = ss_lx.str();
        str_lx = str_lx.substr(0, str_lx.length()-2);
        throw_with_line_number("new lx (" + str_lx + ") must be positive numbers");
    }

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
    if (this->mask != nullptr)
        for(int i=0; i<total_grid; i++)
            dv[i] *= this->mask[i];
}

template <typename T>
void ComputationBox<T>::set_lattice_parameters(std::vector<double> new_lx, std::vector<double> new_angles)
{
    if ( new_lx.size() != (unsigned int) dim )
        throw_with_line_number("The sizes of new lx (" + std::to_string(new_lx.size()) + ") and dim (" + std::to_string(dim) + ") must match.");

    if (std::any_of(new_lx.begin(), new_lx.end(), [](double lx) { return lx <= 0.0;})){
        std::stringstream ss_lx;
        std::copy(new_lx.begin(), new_lx.end(), std::ostream_iterator<int>(ss_lx, ", "));
        std::string str_lx = ss_lx.str();
        str_lx = str_lx.substr(0, str_lx.length()-2);
        throw_with_line_number("new lx (" + str_lx + ") must be positive numbers");
    }

    // Validate angles
    if (new_angles.size() != 3)
        throw_with_line_number("angles must have exactly 3 elements [alpha, beta, gamma], but got " + std::to_string(new_angles.size()));

    for (int i = 0; i < 3; i++) {
        if (new_angles[i] <= 0.0 || new_angles[i] >= 180.0)
            throw_with_line_number("angles[" + std::to_string(i) + "] = " + std::to_string(new_angles[i]) +
                                   " is invalid. Angles must be in range (0, 180) degrees.");
    }

    lx = new_lx;

    // Convert angles from degrees to radians and store
    const double deg_to_rad = std::numbers::pi / 180.0;
    for (int d = 0; d < 3; d++)
        angles_[d] = new_angles[d] * deg_to_rad;

    // Check if orthogonal
    const double angle_tolerance = 1e-10;
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
    if (this->mask != nullptr)
        for(int i=0; i<total_grid; i++)
            dv[i] *= this->mask[i];
}
//-----------------------------------------------------------
// This method calculates integral of g
template <typename T>
T ComputationBox<T>::integral(const T *g)
{
    T sum{0.0};
    for(int i=0; i<total_grid; i++)
        sum += dv[i]*g[i];
    return sum;
}
// This method calculates inner product g and h
template <typename T>
T ComputationBox<T>::inner_product(const T *g, const T *h)
{
    T sum{0.0};
    for(int i=0; i<total_grid; i++)
        sum += dv[i]*g[i]*h[i];
    return sum;
}
// This method calculates inner product g and h with weight 1/w
template <typename T>
T ComputationBox<T>::inner_product_inverse_weight(const T *g, const T *h, const T *w)
{
    T sum{0.0};
    for(int i=0; i<total_grid; i++)
        sum += dv[i]*g[i]*h[i]/w[i];
    return sum;
}
//-----------------------------------------------------------
template <typename T>
T ComputationBox<T>::multi_inner_product(int n_comp, const T *g, const T *h)
{
    T sum{0.0};
    for(int n=0; n < n_comp; n++)
    {
        for(int i=0; i<total_grid; i++)
            sum += dv[i]*g[i+n*total_grid]*h[i+n*total_grid];
    }
    return sum;
}
//-----------------------------------------------------------
// This method makes the input a zero-meaned matrix
template <typename T>
void ComputationBox<T>::zero_mean(T *g)
{
    T sum{0.0};
    for(int i=0; i<total_grid; i++)
        sum += dv[i]*g[i];
    for(int i=0; i<total_grid; i++)
        g[i] -= sum/volume;
}

// Explicit template instantiation
template class ComputationBox<double>;
template class ComputationBox<std::complex<double>>;