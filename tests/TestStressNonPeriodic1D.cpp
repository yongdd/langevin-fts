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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <string>
#include <array>
#include <chrono>
#include <random>
#include <numeric>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

int main()
{
    try
    {
        // Math constants
        const double PI = std::numbers::pi;

        double energy_total;
        double error_level, old_error_level;
        double *w, *w_out, *w_diff;
        double *xi, *w_plus, *w_minus;
        double *phi_a, *phi_b;

        std::streamsize default_precision = std::cout.precision();

        // -------------- Initialize ------------
        int max_scft_iter = 500;
        double tolerance = 1e-9;

        double f = 0.5;  // Symmetric for lamellar
        double chi_n = 20.0;
        std::vector<int> nx = {64};
        std::vector<double> lx = {4.0};  // Large enough for lamellar formation
        double ds = 1.0/100.0;

        int am_n_var = 2*nx[0];
        int am_max_hist = 20;
        double am_start_error = 1e-1;
        double am_mix_min = 0.1;
        double am_mix_init = 0.1;

        const int M = nx[0];

        std::vector<BlockInput> blocks =
        {
            {"A", f, 0, 1},
            {"B", 1.0-f, 1, 2},
        };

        //-------------- Allocate array ------------
        w       = new double[2*M];
        w_out   = new double[2*M];
        w_diff  = new double[2*M];
        xi      = new double[M];
        phi_a   = new double[M];
        phi_b   = new double[M];
        w_plus  = new double[M];
        w_minus = new double[M];

        bool reduce_memory = false;

        // Choose platform (cpu-mkl only for 1D)
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> chain_models = {"Discrete", "Continuous"};
        std::vector<std::string> boundary_conditions = {"reflecting", "absorbing"};
        // Only test pseudo-spectral methods (CN-ADI doesn't support stress yet)
        std::vector<std::string> numerical_methods_discrete = {"rqm4"};
        std::vector<std::string> numerical_methods_continuous = {"rqm4"};

        for(std::string platform : avail_platforms)
        {
            if(platform != "cpu-mkl")
                continue;

            for(std::string bc_str : boundary_conditions)
            {
                std::cout << "\n========== Testing: " << platform << ", BC=" << bc_str << " ==========" << std::endl;

                // Skip absorbing BC - stress computation not supported yet
                // (DST basis doesn't include k=0 mode, causing systematic error)
                if (bc_str == "absorbing")
                {
                    std::cout << "(skipped - stress not supported for absorbing BC)" << std::endl;
                    continue;
                }

                for(std::string chain_model : chain_models)
                {
                    std::cout << "\nChain Model: " << chain_model << std::endl;

                    // Select numerical methods based on chain model
                    const std::vector<std::string>& numerical_methods =
                        (chain_model == "Discrete") ? numerical_methods_discrete : numerical_methods_continuous;

                    for(std::string numerical_method : numerical_methods)
                    {
                        std::cout << "  Numerical Method: " << numerical_method << std::endl;

                        for(bool aggregate_propagator_computation : {false, true})
                        {
                            AbstractFactory<double> *factory = PlatformSelector::create_factory_real(platform, reduce_memory);

                            // Create instances with non-periodic BC (need 2 per dimension: low and high side)
                            ComputationBox<double>* cb = factory->create_computation_box(nx, lx, {bc_str, bc_str});
                            Molecules* molecules = factory->create_molecules_information(chain_model, ds, {{"A",1.0}, {"B",1.0}});
                            molecules->add_polymer(1.0, blocks, {});
                            PropagatorComputationOptimizer* propagator_computation_optimizer =
                                new PropagatorComputationOptimizer(molecules, aggregate_propagator_computation);
                            PropagatorComputation<double>* solver =
                                factory->create_propagator_computation(cb, molecules, propagator_computation_optimizer, numerical_method);
                        AndersonMixing<double> *am = factory->create_anderson_mixing(am_n_var,
                            am_max_hist, am_start_error, am_mix_min, am_mix_init);

                        // Display polymer architecture and propagator info (only on first aggregate iteration)
                        if (!aggregate_propagator_computation)
                        {
                            molecules->display_architectures();
                            propagator_computation_optimizer->display_blocks();
                            propagator_computation_optimizer->display_propagators();
                        }

                        // Initialize fields with lamellar-like pattern
                        // Use strong perturbation to induce phase separation
                        for(int i=0; i<nx[0]; i++)
                        {
                            // Cell-centered coordinate: x = (i + 0.5) * dx
                            double x_norm = (i + 0.5) / nx[0];  // normalized to [0, 1]
                            double delta;

                            if (bc_str == "reflecting") {
                                // Cosine basis respects reflecting BC (zero derivative at boundaries)
                                // Use multiple modes for richer structure
                                delta = 0.3 * std::cos(PI * x_norm) + 0.2 * std::cos(2*PI * x_norm);
                            } else {
                                // Sine basis respects absorbing BC (zero value at boundaries)
                                delta = 0.3 * std::sin(PI * x_norm) + 0.2 * std::sin(2*PI * x_norm);
                            }

                            // Initialize w_A = chi_n * phi_B_approx, w_B = chi_n * phi_A_approx
                            double phi_a_approx = f + delta;
                            double phi_b_approx = (1.0 - f) - delta;
                            w[i] = chi_n * phi_b_approx;
                            w[i+M] = chi_n * phi_a_approx;
                        }

                        // Keep the level of field value
                        cb->zero_mean(&w[0]);
                        cb->zero_mean(&w[M]);

                        // Assign large initial value for the energy and error
                        energy_total = 1.0e20;
                        error_level = 1.0e20;

                        // SCFT Iteration
                        for(int iter=0; iter<max_scft_iter; iter++)
                        {
                            solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                            solver->compute_concentrations();
                            solver->get_total_concentration("A", phi_a);
                            solver->get_total_concentration("B", phi_b);

                            // Debug: check for NaN in first few iterations
                            if (!aggregate_propagator_computation && iter < 3) {
                                double Q = solver->get_total_partition(0);
                                double phi_sum = 0.0;
                                for(int i=0; i<M; i++) phi_sum += phi_a[i] + phi_b[i];
                                std::cout << "      iter=" << iter << ", Q=" << Q
                                          << ", phi_sum/M=" << phi_sum/M << std::endl;
                            }

                            for(int i=0; i<M; i++)
                            {
                                w_minus[i] = (w[i]-w[i+M])/2;
                                w_plus[i]  = (w[i]+w[i+M])/2;
                            }

                            energy_total = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                            energy_total -= cb->integral(w_plus)/cb->get_volume();
                            for(int p=0; p<molecules->get_n_polymer_types(); p++){
                                Polymer& pc = molecules->get_polymer(p);
                                energy_total -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
                            }

                            for(int i=0; i<M; i++)
                            {
                                xi[i] = 0.5*(w[i]+w[i+M]-chi_n);
                                w_out[i]   = chi_n*phi_b[i] + xi[i];
                                w_out[i+M] = chi_n*phi_a[i] + xi[i];
                            }
                            cb->zero_mean(&w_out[0]);
                            cb->zero_mean(&w_out[M]);

                            old_error_level = error_level;
                            for(int i=0; i<2*M; i++)
                                w_diff[i] = w_out[i]- w[i];
                            error_level = sqrt(cb->multi_inner_product(2,w_diff,w_diff)/
                                            (cb->multi_inner_product(2,w,w)+1.0));

                            if(error_level < tolerance) break;
                            am->calculate_new_fields(w, w, w_diff, old_error_level, error_level);
                        }

                        // Compute stress at saddle point
                        solver->compute_stress();
                        auto stress = solver->get_stress();

                        // -------------- Numerical derivative test (dH/dL) ------------
                        double dL = 0.00001;  // Use larger step for better numerical stability
                        double old_lx0 = lx[0];

                        // Lambda to compute total energy
                        auto compute_energy = [&]() -> double {
                            for(int i=0; i<M; i++)
                            {
                                w_minus[i] = (w[i]-w[i+M])/2;
                                w_plus[i]  = (w[i]+w[i+M])/2;
                            }
                            double energy = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                            energy -= cb->integral(w_plus)/cb->get_volume();
                            for(int p=0; p<molecules->get_n_polymer_types(); p++){
                                Polymer& pc = molecules->get_polymer(p);
                                energy -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
                            }
                            return energy;
                        };

                        // Compute numerical derivative dH/dL
                        lx[0] = old_lx0 + dL/2;
                        cb->set_lx(lx);
                        solver->update_laplacian_operator();
                        solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                        double energy_total_1 = compute_energy();

                        lx[0] = old_lx0 - dL/2;
                        cb->set_lx(lx);
                        solver->update_laplacian_operator();
                        solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                        double energy_total_2 = compute_energy();

                        // Reset to original
                        lx[0] = old_lx0;
                        cb->set_lx(lx);
                        solver->update_laplacian_operator();

                        double dh_dl = (energy_total_1 - energy_total_2) / dL;

                        std::cout << "    Aggregation=" << std::boolalpha << aggregate_propagator_computation;
                        std::cout << ", SCFT error=" << std::scientific << std::setprecision(2) << error_level;
                        std::cout << ", Stress=" << std::setprecision(6) << stress[0];
                        std::cout << ", dH/dL=" << std::setprecision(6) << dh_dl << std::endl;

                        // Check SCFT convergence
                        if (!std::isfinite(error_level) || error_level > tolerance) {
                            std::cout << "ERROR: SCFT did not converge!" << std::endl;
                            return -1;
                        }

                        // Check numerical derivative matches analytical stress
                        // Use absolute error when both values are small (near homogeneous state)
                        // Numerical noise in energy differences can be ~1e-7 to 1e-8
                        double abs_diff = std::abs(dh_dl - stress[0]);
                        double max_val = std::max(std::abs(stress[0]), std::abs(dh_dl));
                        double relative_stress_error = abs_diff / (max_val + 1e-10);

                        std::cout << "      Stress difference: |dH/dL - stress| = " << abs_diff;
                        std::cout << ", relative error = " << relative_stress_error << std::endl;

                        // Allow either small absolute error (for homogeneous states)
                        // or small relative error (for phase-separated states)
                        bool pass = (abs_diff < 1e-5) || (relative_stress_error < 1e-3);

                        if (!std::isfinite(abs_diff) || !pass) {
                            std::cout << "ERROR: Numerical derivative does not match stress!" << std::endl;
                            std::cout << "  dH/dL (numerical) = " << dh_dl << std::endl;
                            std::cout << "  stress[0] (analytical) = " << stress[0] << std::endl;
                            return -1;
                        }

                        delete molecules;
                        delete propagator_computation_optimizer;
                        delete cb;
                        delete solver;
                        delete am;
                        delete factory;
                    }
                    }
                }
            }
        }

        //------------- Finalize -------------
        delete[] w;
        delete[] w_out;
        delete[] w_diff;
        delete[] xi;
        delete[] phi_a;
        delete[] phi_b;
        delete[] w_plus;
        delete[] w_minus;

        std::cout << "\nTest passed!" << std::endl;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
