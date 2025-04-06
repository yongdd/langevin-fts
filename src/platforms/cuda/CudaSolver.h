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

template <typename T>
class CudaSolver
{
public:
    // Arrays for real-space method with operator spliting
    std::map<std::string, CuDeviceData<T>*> d_exp_dw;       // Boltzmann factor for the single segment
    std::map<std::string, CuDeviceData<T>*> d_exp_dw_half;  // Boltzmann factor for the half segment

    // CudaSolver(ComputationBox<double>* cb, Molecules *molecules);
    virtual ~CudaSolver() {};
    virtual void update_laplacian_operator() = 0;
    virtual void update_dw(std::string device, std::map<std::string, const T*> w_input) = 0;

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    virtual void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask) = 0;

    // Advance propagator by half bond step
    virtual void advance_propagator_half_bond_step(
        const int STREAM,
        CuDeviceData<T> *q_in, CuDeviceData<T> *q_out, std::string monomer_type) = 0;

    // Compute stress of single segment
    virtual std::vector<T> compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_pair, std::string monomer_type, bool is_half_bond_length) = 0;

//     virtual void compute_single_segment_stress_fourier(const int GPU, double *d_q) = 0;
//     virtual std::vector<double> compute_single_segment_stress_continuous(const int GPU, std::string monomer_type) = 0;
};
#endif
