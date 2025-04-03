/*----------------------------------------------------------
* This class defines a class for real-space method
*-----------------------------------------------------------*/

#ifndef CPU_SOLVER_REAL_SPACE_H_
#define CPU_SOLVER_REAL_SPACE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "FiniteDifference.h"

class CpuSolverRealSpace : public CpuSolver<double>
{
private:
    ComputationBox* cb;
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
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type);
    void advance_propagator_2d(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type);
    void advance_propagator_1d(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type);
public:

    CpuSolverRealSpace(ComputationBox* cb, Molecules *molecules);
    ~CpuSolverRealSpace();
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
    void advance_propagator(
                double *q_in, double *q_out, std::string monomer_type, const double *q_mask) override;

    // Advance propagator by half bond step
    void advance_propagator_half_bond_step(double *q_in, double *q_out, std::string monomer_type) override {};

    // Compute stress of single segment
    std::vector<double> compute_single_segment_stress(
                double *q_1, double *q_2, std::string monomer_type, bool is_half_bond_length) override;
};
#endif
