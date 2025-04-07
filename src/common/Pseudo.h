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
protected:
    std::vector<BoundaryCondition> bc;
    std::map<std::string, double> bond_lengths;
    std::vector<int> nx;
    std::vector<double> dx;
    double ds;

    // Mapping array for negative frequency
    int *k_idx;

    // For stress calculation: compute_stress()
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    // Arrays for pseudo-spectral
    std::map<std::string, double*> boltz_bond;        // Boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // Boltzmann factor for the half bond

    int total_complex_grid;

    void update_total_complex_grid();
    void update_boltz_bond();
    void update_weighted_fourier_basis();
    void update_negative_frequency_mapping();
public:
    Pseudo(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc, std::vector<int> nx, std::vector<double> dx, double ds);
    virtual ~Pseudo();

    int get_total_complex_grid();

    virtual double* get_boltz_bond(std::string monomer_type);
    virtual double* get_boltz_bond_half(std::string monomer_type);

    virtual double* get_fourier_basis_x();
    virtual double* get_fourier_basis_y();
    virtual double* get_fourier_basis_z();

    int* get_negative_frequency_mapping();

    virtual void update(
        std::vector<BoundaryCondition> bc, std::map<std::string, double> bond_lengths,
        std::vector<int> nx, std::vector<double> dx, double ds)
    {
        this->bond_lengths = bond_lengths;
        this->bc = bc;
        this->nx = nx;
        this->dx = dx;
        this->ds = ds;

        update_total_complex_grid();
        update_boltz_bond();
        update_weighted_fourier_basis();
        // update_negative_frequency_mapping();
    };
};
#endif
