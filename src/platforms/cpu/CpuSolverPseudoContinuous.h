/*----------------------------------------------------------
* This class defines a class for pseudo-spectral method for continuous chain
*-----------------------------------------------------------*/

#ifndef CPU_SOLVER_PSEUDO_CONTINUOUS_H_
#define CPU_SOLVER_PSEUDO_CONTINUOUS_H_

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
class CpuSolverPseudoContinuous : public CpuSolver<T>
{
private:
    ComputationBox<T>* cb;
    Molecules *molecules;
    std::string chain_model;

    FFT<T> *fft;
    Pseudo<T> *pseudo;

public:
    CpuSolverPseudoContinuous(ComputationBox<T>* cb, Molecules *molecules);
    ~CpuSolverPseudoContinuous();
    void update_laplacian_operator() override;
    void update_dw(std::map<std::string, const T*> w_input) override;

    // Advance propagator by one contour step
    void advance_propagator(T *q_in, T *q_out, std::string monomer_type, const double *q_mask) override;
    
    // Advance propagator by half bond step
    void advance_propagator_half_bond_step(T *, T *, std::string) override {};
               
    // Compute stress of single segment
    std::vector<T> compute_single_segment_stress(
                T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length) override;
};
#endif
