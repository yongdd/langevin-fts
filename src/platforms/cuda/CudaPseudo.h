/*----------------------------------------------------------
* This class contains static methods for pseudo-spectral Method
*-----------------------------------------------------------*/

#ifndef CUDA_PSEUDO_H_
#define CUDA_PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Pseudo.h"
#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

template <typename T>
class CudaPseudo : public Pseudo<T>
{
private:
    // For stress calculation: compute_stress()
    double *d_fourier_basis_x;
    double *d_fourier_basis_y;
    double *d_fourier_basis_z;
    
    // Mapping array for negative frequency
    int *d_k_idx;

    // GPU arrays for pseudo-spectral
    std::map<std::string, double*> d_boltz_bond;        // Boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half;   // Boltzmann factor for the half bond

    // void update_boltz_bond() override;
    // void update_weighted_fourier_basis() override;
public:
    CudaPseudo(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc, std::vector<int> nx, std::vector<double> dx, double ds);
    ~CudaPseudo();

    double* get_boltz_bond     (std::string monomer_type) override { return d_boltz_bond[monomer_type]; };
    double* get_boltz_bond_half(std::string monomer_type) override { return d_boltz_bond_half[monomer_type];};
    
    double* get_fourier_basis_x() override { return d_fourier_basis_x;};
    double* get_fourier_basis_y() override { return d_fourier_basis_y;};
    double* get_fourier_basis_z() override { return d_fourier_basis_z;};

    void update(
        std::vector<BoundaryCondition> bc, std::map<std::string, double> bond_lengths,
        std::vector<int> nx, std::vector<double> dx, double ds) override;
};
#endif