/*----------------------------------------------------------
* This class contains methods and attributes for pseudo-spectral Method
*-----------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "ComputationBox.h"

class Pseudo
{
private:
    ComputationBox *cb;
    int n_complex_grid;

    void get_boltz_bond(double *boltz_bond, double bond_length_variance,
        std::vector<int> nx, std::vector<double> dx, double ds);
    void get_weighted_fourier_basis(
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::vector<int> nx, std::vector<double> dx);

public:
    Pseudo(ComputationBox *cb);
    virtual ~Pseudo() {};
    virtual void update_bond_function() = 0;
};
#endif
