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
    static void get_boltz_bond(double *boltz_bond, double bond_length_variance,
        std::vector<int> nx, std::vector<double> dx, double ds);
    static void get_weighted_fourier_basis(
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::vector<int> nx, std::vector<double> dx);
    static int get_n_complex_grid(std::vector<int> dx);
};
#endif
