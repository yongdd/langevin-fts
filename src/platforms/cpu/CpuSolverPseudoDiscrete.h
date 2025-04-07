/*----------------------------------------------------------
* This class defines a class for pseudo-spectral method
*-----------------------------------------------------------*/

#ifndef CPU_SOLVER_PSEUDO_DISCRETE_H_
#define CPU_SOLVER_PSEUDO_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Pseudo.h"
#include "FFT.h"

template <typename T>
class CpuSolverPseudoDiscrete : public CpuSolver<T>
{
private:
    ComputationBox<T>* cb;
    Molecules *molecules;
    std::string chain_model;

    FFT<T> *fft;
    Pseudo<T> *pseudo;

public:
    CpuSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules);
    ~CpuSolverPseudoDiscrete();
    void update_laplacian_operator() override;
    void update_dw(std::map<std::string, const T*> w_input) override;

    // Advance propagator by one segment step
    void advance_propagator(T *q_in, T *q_out, std::string monomer_type, const double* q_mask) override;

    // Advance propagator by half bond step
    void advance_propagator_half_bond_step(T *q_in, T *q_out, std::string monomer_type) override;

    // Compute stress of single segment
    std::vector<T> compute_single_segment_stress(
                T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length) override;
};
#endif
