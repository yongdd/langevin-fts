/*----------------------------------------------------------
* This class defines an abstract class for solving propagators
*-----------------------------------------------------------*/

#ifndef CPU_SOLVER_H_
#define CPU_SOLVER_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "FFT.h"

template <typename T>
class CpuSolver
{
public:
    // Arrays for real-space method with operator spliting
    std::map<std::string, T*> exp_dw;            // Boltzmann factor for the single segment
    std::map<std::string, T*> exp_dw_half;       // Boltzmann factor for the half segment

    // CpuSolver(ComputationBox<double>* cb, Molecules *molecules);
    virtual ~CpuSolver() {};
    virtual void update_laplacian_operator() = 0;
    virtual void update_dw(std::map<std::string, const T*> w_input) = 0;

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    virtual void advance_propagator(
        T *q_in, T *q_out, std::string monomer_type, const double *q_mask) = 0;
    
    // Advance propagator by half bond step
    virtual void advance_propagator_half_bond_step(T *q_in, T *q_out, std::string monomer_type) = 0;
                
    // Compute stress of single segment
    virtual std::vector<T> compute_single_segment_stress(
        T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length) = 0;
};
#endif
