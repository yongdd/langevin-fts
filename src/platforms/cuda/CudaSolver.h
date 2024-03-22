/*----------------------------------------------------------
* This class defines an abstract class for solving propagators
*-----------------------------------------------------------*/

#ifndef CUDA_SOLVER_H_
#define CUDA_SOLVER_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CudaCommon.h"

class CudaSolver
{
public:
    // Arrays for real-space method with operator spliting
    std::map<std::string, double*> d_exp_dw[MAX_GPUS];       // Boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half[MAX_GPUS];  // Boltzmann factor for the half segment

    // CudaSolver(ComputationBox *cb, Molecules *molecules);
    virtual ~CudaSolver() {};
    virtual void update_laplacian_operator() = 0;
    virtual void update_dw(std::string device, std::map<std::string, const double*> d_w_input) = 0;

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    virtual void advance_propagator_continuous(
        const int GPU, const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask) = 0;

    virtual void compute_single_segment_stress_continuous(
        const int GPU, const int STREAM,
        double *d_q_pair, double *d_segment_stress, std::string monomer_type) = 0;

//     virtual void compute_single_segment_stress_fourier(const int GPU, double *d_q) = 0;
//     virtual std::vector<double> compute_single_segment_stress_continuous(const int GPU, std::string monomer_type) = 0;
};
#endif
