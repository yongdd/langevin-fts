/*----------------------------------------------------------
* This class defines a class for pseudo-spectral method
*-----------------------------------------------------------*/

#ifndef CPU_SOLVER_PSEUDO_H_
#define CPU_SOLVER_PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuSolverPseudo : public CpuSolver
{
private:
    ComputationBox *cb;
    Molecules *molecules;
    
    FFT *fft;
    std::string chain_model;

    // For stress calculation: compute_stress()
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

public:
    // Arrays for pseudo-spectral
    std::map<std::string, double*> boltz_bond;        // Boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // Boltzmann factor for the half bond

    CpuSolverPseudo(ComputationBox *cb, Molecules *molecules);
    ~CpuSolverPseudo();
    void update_laplacian_operator() override;
    void update_dw(std::map<std::string, const double*> w_input) override;

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    void advance_propagator_continuous(
                double *q_in, double *q_out, std::string monomer_type, const double *q_mask) override;
    
    // Compute stress of single segment
    std::vector<double> compute_single_segment_stress_continuous(
                double *q_1, double *q_2, std::string monomer_type) override;

    //---------- Discrete chain model -------------
    // Advance propagator by one segment step
    void advance_propagator_discrete(double *q_in, double *q_out, std::string monomer_type, const double* q_mask);

    // Advance propagator by half bond step
    void advance_propagator_discrete_half_bond_step(double *q_in, double *q_out, std::string monomer_type);

    // Compute stress of single segment
    std::vector<double> compute_single_segment_stress_discrete(
                double *q_1, double *q_2, std::string monomer_type, bool is_half_bond_length);
};
#endif
