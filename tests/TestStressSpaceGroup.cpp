/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

// Test: stress computed with a SpaceGroup (reduced basis) must agree with
// 1) stress computed WITHOUT a space group on the same exactly-symmetrized
//    full-grid field (rel. tol 1e-6), and
// 2) central finite differences of H = -sum_p (phi_p/alpha_p) ln Q_p with
//    respect to the box lengths (rel. tol 1e-3).
//
// Case A "Pmmm": orthorhombic, unequal box lengths; per-direction FD is valid.
// Case B "Im-3m" (BCC): cubic; the reduced basis enforces cubic symmetry, so
//    per-direction FD is INVALID -- only an isotropic perturbation (all three
//    lx together) is used, compared against the sum of the diagonal stress.
//    In addition, the three diagonal stress components must be equal to each
//    other within 1e-8 (relative).
// Case C "Im-3m" 40^3: same checks as B on a grid the Recursive3m selector
//    historically rejected (nz/2 = 20 not a multiple of 8).
// Case D "P4/mmm": tetragonal; joint Lx = Ly FD vs stress[0]+stress[1] plus
//    Lz-only FD, and stress[0] = stress[1] within 1e-8. Exercises the
//    tetragonal x,y axis-averaging of the CrysFFT stress path (Recursive3m).
// Case E "Im-3m" 12^3: same checks as B; nz/2 = 6 < 8 forces the PmmmDct
//    engine, exercising the cubic axis-averaging of the PmmmDct stress path.
//
// On CPU this exercises the CrysFFT fast path; on CUDA the expand+FFT path.

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "SpaceGroup.h"

namespace
{

struct CaseSpec
{
    std::string name;
    std::vector<int> nx;
    std::vector<double> lx;
    SpaceGroup* sg;
    std::vector<double> w_full;   // exactly symmetrized fields on full grid, size 2*M (A then B)
    std::vector<double> w_red;    // fields in reduced basis, size 2*n_red (A then B)
    bool cubic;                   // true: isotropic FD + component-equality check
    bool tetragonal = false;      // true: joint Lx=Ly FD + x/y component-equality check
};

// H = -sum_p (phi_p/alpha_p) ln Q_p
// (Field terms of the Hamiltonian are volume-normalized and do not depend on lx,
//  so -lnQ suffices for dH/dL finite differences.)
double compute_energy(PropagatorComputation<double>* solver, Molecules* molecules)
{
    double energy = 0.0;
    for(int p=0; p<molecules->get_n_polymer_types(); p++)
    {
        Polymer& pc = molecules->get_polymer(p);
        energy -= pc.get_volume_fraction()/pc.get_alpha()*std::log(solver->get_total_partition(p));
    }
    return energy;
}

// Returns 0 on success, -1 on check failure. Throws on setup errors
// (caught by the caller and treated as "combination not supported").
int run_case(const std::string& platform, const std::string& chain_model, const CaseSpec& cs)
{
    const double f = 0.36;
    const double ds = 1.0/100;
    const int M = cs.nx[0]*cs.nx[1]*cs.nx[2];
    const int n_red = cs.sg->get_n_reduced_basis();

    std::vector<BlockInput> blocks =
    {
        {"A",    f, 0, 1},
        {"B",1.0-f, 1, 2},
    };

    std::cout << "---- Case: " << cs.name << " (" << platform << ", " << chain_model << ") ----" << std::endl;
    std::cout << "n_reduced_basis: " << n_red << " / " << M << std::endl;

    AbstractFactory<double>* factory = PlatformSelector::create_factory_real(platform, false);

    // Solver 1: WITH space group; fields are passed in the reduced basis
    ComputationBox<double>* cb_sg  = factory->create_computation_box(cs.nx, cs.lx, {});
    Molecules* molecules_sg        = factory->create_molecules_information(chain_model, ds, {{"A",1.0}, {"B",1.0}});
    molecules_sg->add_polymer(1.0, blocks, {});
    PropagatorComputationOptimizer* optimizer_sg = new PropagatorComputationOptimizer(molecules_sg, false);
    PropagatorComputation<double>* solver_sg = factory->create_propagator_computation(
        cb_sg, molecules_sg, optimizer_sg, "rqm4", cs.sg);

    // Solver 2: WITHOUT space group; fed the same symmetrized full-grid fields
    ComputationBox<double>* cb_std = factory->create_computation_box(cs.nx, cs.lx, {});
    Molecules* molecules_std       = factory->create_molecules_information(chain_model, ds, {{"A",1.0}, {"B",1.0}});
    molecules_std->add_polymer(1.0, blocks, {});
    PropagatorComputationOptimizer* optimizer_std = new PropagatorComputationOptimizer(molecules_std, false);
    PropagatorComputation<double>* solver_std = factory->create_propagator_computation(
        cb_std, molecules_std, optimizer_std, "rqm4");

    std::map<std::string, const double*> w_map_sg =
        {{"A", cs.w_red.data()}, {"B", cs.w_red.data()+n_red}};
    std::map<std::string, const double*> w_map_std =
        {{"A", cs.w_full.data()}, {"B", cs.w_full.data()+M}};

    // -------- Stress with space group (reduced basis) --------
    solver_sg->compute_propagators(w_map_sg, {});
    solver_sg->compute_stress();
    std::vector<double> stress_sg = solver_sg->get_stress();

    // -------- Stress without space group (full grid) --------
    solver_std->compute_propagators(w_map_std, {});
    solver_std->compute_stress();
    std::vector<double> stress_std = solver_std->get_stress();

    std::cout << std::setprecision(12);
    for(int d=0; d<3; d++)
    {
        std::cout << "d=" << d << " stress_sg=" << stress_sg[d]
                  << " stress_std=" << stress_std[d] << std::endl;
    }

    // ============ CHECK 1: space-group stress vs standard stress ============
    for(int d=0; d<3; d++)
    {
        double rel = std::abs(stress_sg[d]-stress_std[d])/std::abs(stress_std[d]);
        std::cout << "d=" << d << " |stress_sg - stress_std|/|stress_std| = " << rel << std::endl;
        if (!std::isfinite(rel) || rel > 1e-6)
        {
            std::cout << "ERROR: space-group stress does not match standard stress." << std::endl;
            return -1;
        }
    }
    if (cs.cubic)
    {
        // Cubic symmetry: the three diagonal components must be equal
        for(int d=1; d<3; d++)
        {
            double rel = std::abs(stress_sg[d]-stress_sg[0])/std::abs(stress_sg[0]);
            std::cout << "d=" << d << " |stress_sg[d] - stress_sg[0]|/|stress_sg[0]| = " << rel << std::endl;
            if (!std::isfinite(rel) || rel > 1e-8)
            {
                std::cout << "ERROR: cubic stress components are not equal." << std::endl;
                return -1;
            }
        }
    }
    if (cs.tetragonal)
    {
        // Tetragonal symmetry: the x and y components must be equal
        double rel = std::abs(stress_sg[1]-stress_sg[0])/std::abs(stress_sg[0]);
        std::cout << "|stress_sg[1] - stress_sg[0]|/|stress_sg[0]| = " << rel << std::endl;
        if (!std::isfinite(rel) || rel > 1e-8)
        {
            std::cout << "ERROR: tetragonal x/y stress components are not equal." << std::endl;
            return -1;
        }
    }

    // ============ CHECK 2: finite-difference dH/dL vs stress ============
    if (cs.tetragonal)
    {
        // The reduced basis enforces x = y, so perturb Lx = Ly jointly
        // (compared against stress[0] + stress[1]) and Lz alone.
        {
            double dl = cs.lx[0]*2e-4;
            std::vector<double> lx_p = cs.lx, lx_m = cs.lx;
            lx_p[0] = cs.lx[0] + dl/2; lx_p[1] = cs.lx[1] + dl/2;
            lx_m[0] = cs.lx[0] - dl/2; lx_m[1] = cs.lx[1] - dl/2;

            cb_sg->set_lx(lx_p);
            solver_sg->update_laplacian_operator();
            solver_sg->compute_propagators(w_map_sg, {});
            double energy_p = compute_energy(solver_sg, molecules_sg);

            cb_sg->set_lx(lx_m);
            solver_sg->update_laplacian_operator();
            solver_sg->compute_propagators(w_map_sg, {});
            double energy_m = compute_energy(solver_sg, molecules_sg);

            double dh_dl = (energy_p-energy_m)/dl;
            double stress_xy = stress_sg[0]+stress_sg[1];
            double rel = std::abs(dh_dl-stress_xy)/std::abs(stress_xy);
            std::cout << "dH/d(Lx=Ly) : " << dh_dl << std::endl;
            std::cout << "stress[0]+stress[1] : " << stress_xy << std::endl;
            std::cout << "Relative stress error : " << rel << std::endl;
            if (!std::isfinite(rel) || rel > 1e-3)
            {
                std::cout << "ERROR: joint Lx=Ly FD does not match x+y stress." << std::endl;
                return -1;
            }
        }
        {
            double dl = cs.lx[2]*2e-4;
            std::vector<double> lx_p = cs.lx, lx_m = cs.lx;
            lx_p[2] = cs.lx[2] + dl/2;
            lx_m[2] = cs.lx[2] - dl/2;

            cb_sg->set_lx(lx_p);
            solver_sg->update_laplacian_operator();
            solver_sg->compute_propagators(w_map_sg, {});
            double energy_p = compute_energy(solver_sg, molecules_sg);

            cb_sg->set_lx(lx_m);
            solver_sg->update_laplacian_operator();
            solver_sg->compute_propagators(w_map_sg, {});
            double energy_m = compute_energy(solver_sg, molecules_sg);

            cb_sg->set_lx(cs.lx);
            solver_sg->update_laplacian_operator();

            double dh_dl = (energy_p-energy_m)/dl;
            double rel = std::abs(dh_dl-stress_sg[2])/std::abs(stress_sg[2]);
            std::cout << "dH/dLz : " << dh_dl << std::endl;
            std::cout << "stress[2] : " << stress_sg[2] << std::endl;
            std::cout << "Relative stress error : " << rel << std::endl;
            if (!std::isfinite(rel) || rel > 1e-3)
            {
                std::cout << "ERROR: Lz FD does not match z stress." << std::endl;
                return -1;
            }
        }
    }
    else if (cs.cubic)
    {
        // Isotropic perturbation ONLY: under a cubic space group the reduced
        // basis enforces cubic symmetry, so per-direction FD is confounded.
        double dl = cs.lx[0]*2e-4;
        std::vector<double> lx_p = cs.lx, lx_m = cs.lx;
        for(int d=0; d<3; d++)
        {
            lx_p[d] = cs.lx[d] + dl/2;
            lx_m[d] = cs.lx[d] - dl/2;
        }

        cb_sg->set_lx(lx_p);
        solver_sg->update_laplacian_operator();
        solver_sg->compute_propagators(w_map_sg, {});
        double energy_p = compute_energy(solver_sg, molecules_sg);

        cb_sg->set_lx(lx_m);
        solver_sg->update_laplacian_operator();
        solver_sg->compute_propagators(w_map_sg, {});
        double energy_m = compute_energy(solver_sg, molecules_sg);

        cb_sg->set_lx(cs.lx);
        solver_sg->update_laplacian_operator();

        double dh_dl_iso = (energy_p-energy_m)/dl;
        double stress_iso = stress_sg[0]+stress_sg[1]+stress_sg[2];
        double rel = std::abs(dh_dl_iso-stress_iso)/std::abs(stress_iso);
        std::cout << "dH/dl_iso : " << dh_dl_iso << std::endl;
        std::cout << "Sum of diagonal stress : " << stress_iso << std::endl;
        std::cout << "Relative stress error : " << rel << std::endl;
        if (!std::isfinite(rel) || rel > 1e-3)
        {
            std::cout << "ERROR: isotropic FD does not match sum of stress components." << std::endl;
            return -1;
        }
    }
    else
    {
        // Pmmm (no axis permutation): per-direction FD is valid
        for(int d=0; d<3; d++)
        {
            double dl = cs.lx[d]*2e-4;

            std::vector<double> lx_p = cs.lx;
            lx_p[d] = cs.lx[d] + dl/2;
            cb_sg->set_lx(lx_p);
            solver_sg->update_laplacian_operator();
            solver_sg->compute_propagators(w_map_sg, {});
            double energy_p = compute_energy(solver_sg, molecules_sg);

            std::vector<double> lx_m = cs.lx;
            lx_m[d] = cs.lx[d] - dl/2;
            cb_sg->set_lx(lx_m);
            solver_sg->update_laplacian_operator();
            solver_sg->compute_propagators(w_map_sg, {});
            double energy_m = compute_energy(solver_sg, molecules_sg);

            cb_sg->set_lx(cs.lx);
            solver_sg->update_laplacian_operator();

            double dh_dl = (energy_p-energy_m)/dl;
            double rel = std::abs(dh_dl-stress_sg[d])/std::abs(stress_sg[d]);
            std::cout << "d=" << d << " dH/dL : " << dh_dl << std::endl;
            std::cout << "d=" << d << " Stress : " << stress_sg[d] << std::endl;
            std::cout << "d=" << d << " Relative stress error : " << rel << std::endl;
            if (!std::isfinite(rel) || rel > 1e-3)
            {
                std::cout << "ERROR: per-direction FD does not match space-group stress." << std::endl;
                return -1;
            }
        }
    }

    delete solver_sg;
    delete optimizer_sg;
    delete molecules_sg;
    delete cb_sg;
    delete solver_std;
    delete optimizer_std;
    delete molecules_std;
    delete cb_std;
    delete factory;

    std::cout << "Case passed." << std::endl;
    return 0;
}

// Exactly symmetrize the two stacked fields (A then B, size 2*M) by
// round-tripping through the space group reduced basis.
void symmetrize_fields(CaseSpec& cs)
{
    const int M = cs.nx[0]*cs.nx[1]*cs.nx[2];
    const int n_red = cs.sg->get_n_reduced_basis();
    cs.w_red.assign(2*n_red, 0.0);
    std::vector<double> w_sym(2*M, 0.0);
    cs.sg->to_reduced_basis(cs.w_full.data(), cs.w_red.data(), 2);
    cs.sg->from_reduced_basis(cs.w_red.data(), w_sym.data(), 2);

    double max_asym = 0.0;
    for(int i=0; i<2*M; i++)
        max_asym = std::max(max_asym, std::abs(w_sym[i]-cs.w_full[i]));
    std::cout << cs.name << ": max |w_sym - w_analytic| = " << max_asym << std::endl;

    cs.w_full = w_sym;
}

} // namespace

int main()
{
    try
    {
        const double PI = std::numbers::pi;

        // ================= Case A: Pmmm, orthorhombic, unequal L =================
        CaseSpec case_a;
        case_a.name = "Pmmm (orthorhombic)";
        case_a.nx = {32, 24, 16};
        case_a.lx = {3.0, 2.0, 1.5};
        case_a.cubic = false;
        SpaceGroup sg_pmmm(case_a.nx, "Pmmm");
        case_a.sg = &sg_pmmm;
        {
            const int M = case_a.nx[0]*case_a.nx[1]*case_a.nx[2];
            case_a.w_full.assign(2*M, 0.0);
            for(int i=0; i<case_a.nx[0]; i++)
            {
                double X = (i+0.5)/case_a.nx[0];  // fractional cell-centered coordinate
                for(int j=0; j<case_a.nx[1]; j++)
                {
                    double Y = (j+0.5)/case_a.nx[1];
                    for(int k=0; k<case_a.nx[2]; k++)
                    {
                        double Z = (k+0.5)/case_a.nx[2];
                        int idx = i*case_a.nx[1]*case_a.nx[2] + j*case_a.nx[2] + k;
                        // Smooth, mirror-symmetric (Pmmm-compatible) fields, std ~ 5
                        case_a.w_full[idx] = 5.0*(std::cos(2*PI*X) + std::cos(2*PI*Y) + std::cos(2*PI*Z)
                                            + 0.7*std::cos(2*PI*X)*std::cos(2*PI*Y)
                                            + 0.5*std::cos(4*PI*Z));
                        case_a.w_full[idx+M] = 5.0*(std::cos(2*PI*Y)*std::cos(2*PI*Z)
                                              + 0.8*std::cos(4*PI*X));
                    }
                }
            }
        }
        symmetrize_fields(case_a);

        // ================= Case B: Im-3m (BCC), cubic =================
        CaseSpec case_b;
        case_b.name = "Im-3m (BCC, cubic)";
        case_b.nx = {32, 32, 32};
        case_b.lx = {1.9, 1.9, 1.9};
        case_b.cubic = true;
        SpaceGroup sg_bcc(case_b.nx, "Im-3m", 529);
        case_b.sg = &sg_bcc;
        {
            const int M = case_b.nx[0]*case_b.nx[1]*case_b.nx[2];
            case_b.w_full.assign(2*M, 0.0);
            for(int i=0; i<case_b.nx[0]; i++)
            {
                double X = (i+0.5)/case_b.nx[0];
                for(int j=0; j<case_b.nx[1]; j++)
                {
                    double Y = (j+0.5)/case_b.nx[1];
                    for(int k=0; k<case_b.nx[2]; k++)
                    {
                        double Z = (k+0.5)/case_b.nx[2];
                        int idx = i*case_b.nx[1]*case_b.nx[2] + j*case_b.nx[2] + k;
                        // Smooth Im-3m-compatible BCC harmonics ({110} + {200}), std ~ 5
                        double c110 = std::cos(2*PI*X)*std::cos(2*PI*Y)
                                    + std::cos(2*PI*Y)*std::cos(2*PI*Z)
                                    + std::cos(2*PI*Z)*std::cos(2*PI*X);
                        double c200 = std::cos(4*PI*X) + std::cos(4*PI*Y) + std::cos(4*PI*Z);
                        double w_a = 5.0*(c110 + 0.4*c200);
                        case_b.w_full[idx]   = w_a;
                        case_b.w_full[idx+M] = -0.6*w_a;
                    }
                }
            }
        }
        symmetrize_fields(case_b);

        // Case C: same cubic group on a 40^3 grid, where nz/2 = 20 is not a
        // multiple of 8. Historically this grid fell back to the PmmmDct engine
        // (catching the MklCrysFFTPmmm DCT-normalization bug); since the
        // Recursive3m grid condition was relaxed to nz/2 >= 8, this now
        // exercises the Recursive3m engine on a non-multiple-of-16 grid that
        // the selector previously rejected outright.
        CaseSpec case_c = case_b;
        case_c.name = "Im-3m (BCC, cubic, 40^3 non-multiple-of-16 grid)";
        case_c.nx = {40, 40, 40};
        SpaceGroup sg_bcc40(case_c.nx, "Im-3m", 529);
        case_c.sg = &sg_bcc40;
        {
            const int M = case_c.nx[0]*case_c.nx[1]*case_c.nx[2];
            case_c.w_full.assign(2*M, 0.0);
            for(int i=0; i<case_c.nx[0]; i++)
            {
                double X = (i+0.5)/case_c.nx[0];
                for(int j=0; j<case_c.nx[1]; j++)
                {
                    double Y = (j+0.5)/case_c.nx[1];
                    for(int k=0; k<case_c.nx[2]; k++)
                    {
                        double Z = (k+0.5)/case_c.nx[2];
                        int idx = i*case_c.nx[1]*case_c.nx[2] + j*case_c.nx[2] + k;
                        double c110 = std::cos(2*PI*X)*std::cos(2*PI*Y)
                                    + std::cos(2*PI*Y)*std::cos(2*PI*Z)
                                    + std::cos(2*PI*Z)*std::cos(2*PI*X);
                        double c200 = std::cos(4*PI*X) + std::cos(4*PI*Y) + std::cos(4*PI*Z);
                        double w_a = 5.0*(c110 + 0.4*c200);
                        case_c.w_full[idx]   = w_a;
                        case_c.w_full[idx+M] = -0.6*w_a;
                    }
                }
            }
        }
        symmetrize_fields(case_c);

        // ================= Case D: P4/mmm, tetragonal =================
        // 4-fold rotations permute the x and y axes, so the per-axis CrysFFT
        // stress multipliers are non-orbit-invariant in x,y and the x,y sums
        // must be averaged (the "tetragonal" mode of symmetrize_axis_sums /
        // sym_mode = 1). 24x24x16 selects the Recursive3m engine.
        CaseSpec case_d;
        case_d.name = "P4/mmm (tetragonal)";
        case_d.nx = {24, 24, 16};
        case_d.lx = {2.0, 2.0, 1.5};
        case_d.cubic = false;
        case_d.tetragonal = true;
        SpaceGroup sg_tet(case_d.nx, "P4/mmm");
        case_d.sg = &sg_tet;
        {
            const int M = case_d.nx[0]*case_d.nx[1]*case_d.nx[2];
            case_d.w_full.assign(2*M, 0.0);
            for(int i=0; i<case_d.nx[0]; i++)
            {
                double X = (i+0.5)/case_d.nx[0];
                for(int j=0; j<case_d.nx[1]; j++)
                {
                    double Y = (j+0.5)/case_d.nx[1];
                    for(int k=0; k<case_d.nx[2]; k++)
                    {
                        double Z = (k+0.5)/case_d.nx[2];
                        int idx = i*case_d.nx[1]*case_d.nx[2] + j*case_d.nx[2] + k;
                        // Smooth P4/mmm-compatible harmonics (x<->y symmetric), std ~ 5
                        double c10 = std::cos(2*PI*X) + std::cos(2*PI*Y);
                        double c11 = std::cos(2*PI*X)*std::cos(2*PI*Y);
                        double cz  = std::cos(2*PI*Z);
                        double w_a = 5.0*(c10 + 0.6*c11 + 0.8*cz + 0.5*c10*cz);
                        case_d.w_full[idx]   = w_a;
                        case_d.w_full[idx+M] = -0.6*w_a + 2.0*c11*cz;
                    }
                }
            }
        }
        symmetrize_fields(case_d);

        // ================= Case E: Im-3m on 12^3 (PmmmDct fallback) =================
        // nz/2 = 6 < 8 fails the Recursive3m grid bound, so the selector falls
        // back to the PmmmDct engine; this exercises the cubic averaging branch
        // of the PmmmDct stress path, which larger cubic grids no longer reach
        // (they select Recursive3m since the nz/2 >= 8 relaxation).
        CaseSpec case_e = case_b;
        case_e.name = "Im-3m (BCC, cubic, 12^3 PmmmDct fallback)";
        case_e.nx = {12, 12, 12};
        case_e.lx = {1.9, 1.9, 1.9};  // same cubic box as case B, coarser grid
        SpaceGroup sg_bcc12(case_e.nx, "Im-3m", 529);
        case_e.sg = &sg_bcc12;
        {
            const int M = case_e.nx[0]*case_e.nx[1]*case_e.nx[2];
            case_e.w_full.assign(2*M, 0.0);
            for(int i=0; i<case_e.nx[0]; i++)
            {
                double X = (i+0.5)/case_e.nx[0];
                for(int j=0; j<case_e.nx[1]; j++)
                {
                    double Y = (j+0.5)/case_e.nx[1];
                    for(int k=0; k<case_e.nx[2]; k++)
                    {
                        double Z = (k+0.5)/case_e.nx[2];
                        int idx = i*case_e.nx[1]*case_e.nx[2] + j*case_e.nx[2] + k;
                        double c110 = std::cos(2*PI*X)*std::cos(2*PI*Y)
                                    + std::cos(2*PI*Y)*std::cos(2*PI*Z)
                                    + std::cos(2*PI*Z)*std::cos(2*PI*X);
                        double c200 = std::cos(4*PI*X) + std::cos(4*PI*Y) + std::cos(4*PI*Z);
                        double w_a = 5.0*(c110 + 0.4*c200);
                        case_e.w_full[idx]   = w_a;
                        case_e.w_full[idx+M] = -0.6*w_a;
                    }
                }
            }
        }
        symmetrize_fields(case_e);

        // ================= Run all platform / chain-model combinations =================
        // Any CPU platform (cpu-mkl or cpu-fftw) MUST run for BOTH chain models:
        // the CPU CrysFFT stress path and the discrete+space-group path are the
        // regressions this test protects. Exceptions on CPU are failures, not
        // skips; only CUDA combinations may be skipped (e.g. discrete+space-group
        // is not supported on CUDA).
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        bool ran_cpu_continuous = false;
        bool ran_cpu_discrete = false;

        for(const std::string& platform : avail_platforms)
        {
            const bool is_cpu = (platform.rfind("cpu", 0) == 0);
            for(const std::string& chain_model : chain_models)
            {
                for(const CaseSpec* cs : {&case_a, &case_b, &case_c, &case_d, &case_e})
                {
                    std::cout << "==============================================" << std::endl;
                    std::cout << "Testing: " << platform << ", " << chain_model
                              << ", " << cs->name << std::endl;
                    try
                    {
                        if (run_case(platform, chain_model, *cs) != 0)
                            return -1;
                    }
                    catch(std::exception& exc)
                    {
                        if (is_cpu)
                        {
                            std::cout << "ERROR: CPU combination (" << platform << ", "
                                      << chain_model << ", " << cs->name << ") threw:" << std::endl;
                            std::cout << exc.what() << std::endl;
                            return -1;
                        }
                        std::cout << "Note: skipping unsupported combination (" << platform
                                  << ", " << chain_model << ", " << cs->name << "):" << std::endl;
                        std::cout << exc.what() << std::endl;
                        continue;
                    }
                }
                if (is_cpu && chain_model == "Continuous")
                    ran_cpu_continuous = true;
                if (is_cpu && chain_model == "Discrete")
                    ran_cpu_discrete = true;
            }
        }

        if (!ran_cpu_continuous || !ran_cpu_discrete)
        {
            std::cout << "ERROR: required CPU combinations did not run "
                      << "(continuous: " << ran_cpu_continuous
                      << ", discrete: " << ran_cpu_discrete << ")." << std::endl;
            return -1;
        }

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
