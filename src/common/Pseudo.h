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

template <typename T>
class Pseudo
{
private:

public:
    static void get_boltz_bond(
        std::vector<BoundaryCondition> bc,
        double *boltz_bond, double bond_length_variance,
        std::vector<int> nx, std::vector<double> dx, double ds);
    static void get_weighted_fourier_basis(
        std::vector<BoundaryCondition> bc,
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::vector<int> nx, std::vector<double> dx);
    static int get_total_complex_grid(std::vector<int> dx);

    static void get_negative_frequency_mapping(std::vector<int> nx, int *k_idx);
};
#endif
