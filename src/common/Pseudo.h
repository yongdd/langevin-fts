/*----------------------------------------------------------
* This class contains static methods for pseudo-spectral Method
*-----------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

class Pseudo
{
public:
    // Real-to-complex transform
    static void get_boltz_bond_r2c(
        std::vector<BoundaryCondition> bc,
        double *boltz_bond, double bond_length_variance,
        std::vector<int> nx, std::vector<double> dx, double ds);
    static void get_weighted_fourier_basis_r2c(
        std::vector<BoundaryCondition> bc,
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::vector<int> nx, std::vector<double> dx);
    static int get_total_complex_grid_r2c(std::vector<int> dx);

    // complex-to-complex transform
    static void get_boltz_bond_c2c(
        std::vector<BoundaryCondition> bc,
        double *boltz_bond, double bond_length_variance,
        std::vector<int> nx, std::vector<double> dx, double ds);
    static void get_weighted_fourier_basis_c2c(
        std::vector<BoundaryCondition> bc,
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::vector<int> nx, std::vector<double> dx);
    static int get_total_complex_grid_c2c(std::vector<int> dx);

    template <typename T>
    static void get_boltz_bond(
        std::vector<BoundaryCondition> bc,
        double *boltz_bond, double bond_length_variance,
        std::vector<int> nx, std::vector<double> dx, double ds)
    {
        if constexpr (std::is_same<T, double>::value)
            get_boltz_bond_r2c(bc, boltz_bond, bond_length_variance, nx, dx, ds);
        else
            get_boltz_bond_c2c(bc, boltz_bond, bond_length_variance, nx, dx, ds);
    }

    template <typename T>
    static void get_weighted_fourier_basis(
        std::vector<BoundaryCondition> bc,
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::vector<int> nx, std::vector<double> dx)
    {
        if constexpr (std::is_same<T, double>::value)
            get_weighted_fourier_basis_r2c(bc, fourier_basis_x, fourier_basis_y, fourier_basis_z, nx, dx);
        else
            get_weighted_fourier_basis_c2c(bc, fourier_basis_x, fourier_basis_y, fourier_basis_z, nx, dx);
    }

    template <typename T>
    static int get_total_complex_grid(std::vector<int> nx)
    {
        if constexpr (std::is_same<T, double>::value)
            return get_total_complex_grid_r2c(nx);
        else
            return get_total_complex_grid_c2c(nx);
    }
};
#endif
