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
    ~CudaSolver() {};
    virtual void update_laplacian_operator() = 0;
    virtual void update_dw(std::string device, std::map<std::string, const double*> d_w_input) = 0;

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    virtual void advance_one_propagator_continuous(const int GPU,
            double *d_q_in, double *d_q_out,
            std::string monomer_type, double *d_q_mask) = 0;

    // Advance two propagators by one contour step
    virtual void advance_two_propagators_continuous(
            double *d_q_in_1,  double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double *d_q_mask) = 0;

    // Advance two propagators by one segment step in two GPUs
    virtual void advance_two_propagators_continuous_two_gpus(
            double *d_q_in_1,  double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double **d_q_mask) = 0;

    virtual void compute_single_segment_stress_fourier(const int GPU, double *d_q) = 0;
    virtual std::vector<double> compute_single_segment_stress_continuous(const int GPU, std::string monomer_type) = 0;
};
#endif
