/*----------------------------------------------------------
* This class defines a class for real-space method
*-----------------------------------------------------------*/

#ifndef CPU_SOLVER_REAL_H_
#define CPU_SOLVER_REAL_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "FiniteDifference.h"

class CpuSolverReal : public CpuSolver
{
private:
    ComputationBox *cb;
    Molecules *molecules;
    
    // Trigonal matrix for x direction
    std::map<std::string, double*> xl;
    std::map<std::string, double*> xd;
    std::map<std::string, double*> xh;

    // Trigonal matrix for y direction
    std::map<std::string, double*> yl;
    std::map<std::string, double*> yd;
    std::map<std::string, double*> yh;

    // Trigonal matrix for z direction
    std::map<std::string, double*> zl;
    std::map<std::string, double*> zd;
    std::map<std::string, double*> zh;

    int max_of_two(int x, int y);
    int min_of_two(int x, int y);

    void advance_propagator_3d(
        double *q_in, double *q_out, std::string monomer_type);
    void advance_propagator_2d(
        double *q_in, double *q_out, std::string monomer_type);
    void advance_propagator_1d(
        double *q_in, double *q_out, std::string monomer_type);
public:

    CpuSolverReal(ComputationBox *cb, Molecules *molecules);
    ~CpuSolverReal();
    void update_laplacian_operator() override;
    void update_dw(std::map<std::string, const double*> w_input) override;

    static void tridiagonal(
        const double *xl, const double *xd, const double *xh,
        double *x, const int INTERVAL, const double *d, const int M);

    static void tridiagonal_periodic(
        const double *xl, const double *xd, const double *xh,
        double *x, const int INTERVAL, const double *d, const int M);

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    void advance_propagator_continuous(
                double *q_in, double *q_out, std::string monomer_type, const double *q_mask) override;
    
    // Compute stress of single segment
    std::vector<double> compute_single_segment_stress_continuous(
                double *q_1, double *q_2, std::string monomer_type) override;
};
#endif
