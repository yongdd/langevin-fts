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

class CpuSolver
{
public:
    // Arrays for real-space method with operator spliting
    std::map<std::string, double*> exp_dw;            // Boltzmann factor for the single segment
    std::map<std::string, double*> exp_dw_half;       // Boltzmann factor for the half segment

    // CpuSolver(ComputationBox *cb, Molecules *molecules);
    virtual ~CpuSolver() {};
    virtual void update_laplacian_operator() = 0;
    virtual void update_dw(std::map<std::string, const double*> w_input) = 0;

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    virtual void advance_propagator(
                double *q_in, double *q_out, std::string monomer_type, const double *q_mask) = 0;
    
    // Advance propagator by half bond step
    virtual void advance_propagator_half_bond_step(double *q_in, double *q_out, std::string monomer_type) = 0;
                
    // Compute stress of single segment
    virtual std::vector<double> compute_single_segment_stress(
                double *q_1, double *q_2, std::string monomer_type, bool is_half_bond_length) = 0;
};
#endif
