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
private:
    ComputationBox *cb;
    Molecules *molecules;
public:
    // Arrays for operator splitting
    std::map<std::string, double*> exp_dw;       // Boltzmann factor for the single segment
    std::map<std::string, double*> exp_dw_half;  // Boltzmann factor for the half segment

    CpuSolver(ComputationBox *cb, Molecules *molecules);
    ~CpuSolver();
    void update_laplacian_operator();
    void update_dw(std::map<std::string, const double*> w_input);

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    void advance_propagator_continuous(
                double *q_in, double *q_out, std::string monomer_type, const double *q_mask);
    
    // Compute stress of single segment
    std::vector<double> compute_single_segment_stress_continuous(
                double *q_1, double *q_2, std::string monomer_type);
};
#endif
