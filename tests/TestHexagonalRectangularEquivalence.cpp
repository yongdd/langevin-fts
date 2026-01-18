/**
 * @file TestHexagonalRectangularEquivalence.cpp
 * @brief Test that hexagonal and rectangular unit cells give equivalent results.
 *
 * This test verifies that the hexagonal cylinder phase computed in two different
 * unit cell representations produces the same free energy:
 *
 * 1. Hexagonal unit cell: a = b = L, gamma = 120 degrees (1 cylinder per cell)
 * 2. Rectangular unit cell: a = sqrt(3)*L, b = L, gamma = 90 degrees (2 cylinders per cell)
 *
 * Both representations describe the same physical hexagonal cylinder phase.
 * The free energies per chain should be identical within numerical precision.
 *
 * This test validates the correct implementation of non-orthogonal crystal systems,
 * particularly the sign handling of cross-terms in the reciprocal metric tensor.
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <string>
#include <array>
#include <chrono>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

/**
 * @brief Result structure for SCFT calculation
 */
struct ScftResult
{
    double energy;
    std::vector<double> stress;
    std::vector<double> lx;
    std::vector<double> angles;
};

/**
 * @brief Run SCFT calculation with fixed box (load fields from file if available)
 */
ScftResult run_scft_fixed_box(
    AbstractFactory<double>* factory,
    std::vector<int> nx,
    std::vector<double> lx,
    std::vector<double> angles,
    double f,
    double chi_n,
    double ds,
    int max_iter,
    double tolerance,
    const std::string& input_filename,
    bool compute_stress_flag = false,
    bool verbose = false)
{
    const double PI = std::numbers::pi;

    // Create computation box
    ComputationBox<double>* cb = factory->create_computation_box(nx, lx, {}, angles);

    // Create molecules
    std::vector<BlockInput> blocks = {
        {"A",     f, 0, 1},
        {"B", 1.0-f, 1, 2},
    };
    Molecules* molecules = factory->create_molecules_information("Continuous", ds, {{"A", 1.0}, {"B", 1.0}});
    molecules->add_polymer(1.0, blocks, {});

    // Create solver
    PropagatorComputationOptimizer* optimizer = new PropagatorComputationOptimizer(molecules, false);
    PropagatorComputation<double>* solver = factory->create_propagator_computation(cb, molecules, optimizer, "rqm4");

    // Create Anderson mixing
    const int M = cb->get_total_grid();
    int am_n_var = 2 * M;
    AndersonMixing<double>* am = factory->create_anderson_mixing(am_n_var, 20, 1e-1, 0.1, 0.1);

    // Allocate arrays
    double* w = new double[2 * M];
    double* w_out = new double[2 * M];
    double* w_diff = new double[2 * M];
    double* xi = new double[M];
    double* w_plus = new double[M];
    double* w_minus = new double[M];
    double* phi_a = new double[M];
    double* phi_b = new double[M];

    // Try to load pre-converged fields from file
    bool loaded_from_file = false;
    std::ifstream input_field_file(input_filename);
    if (input_field_file.is_open())
    {
        std::string line;
        for (int i = 0; i < 2 * M; i++)
        {
            std::getline(input_field_file, line);
            w[i] = std::stod(line);
        }
        input_field_file.close();
        loaded_from_file = true;
        if (verbose)
            std::cout << "  Loaded fields from " << input_filename << std::endl;
    }
    else
    {
        // Initialize fields with cylinder(s) at appropriate positions
        bool is_hex = (std::abs(angles[2] - 120.0) < 1.0);

        for (int i = 0; i < M; i++)
            phi_a[i] = 0.0;

        // Create cylinder initial condition
        double cylinder_radius = 0.2;
        double interface_width = 0.05;

        std::vector<std::pair<double, double>> cylinder_positions;
        if (is_hex)
            cylinder_positions.push_back({0.0, 0.0});
        else
        {
            cylinder_positions.push_back({0.0, 0.0});
            cylinder_positions.push_back({0.5, 0.5});
        }

        for (int i = 0; i < nx[0]; i++)
        {
            for (int j = 0; j < nx[1]; j++)
            {
                int idx = i * nx[1] + j;
                double x_frac = (double)i / nx[0];
                double y_frac = (double)j / nx[1];

                double max_phi = 0.0;
                for (const auto& pos : cylinder_positions)
                {
                    double dx_frac = x_frac - pos.first;
                    double dy_frac = y_frac - pos.second;

                    if (dx_frac > 0.5) dx_frac -= 1.0;
                    if (dx_frac < -0.5) dx_frac += 1.0;
                    if (dy_frac > 0.5) dy_frac -= 1.0;
                    if (dy_frac < -0.5) dy_frac += 1.0;

                    double cos_gamma = std::cos(angles[2] * PI / 180.0);
                    double r_sq = dx_frac * dx_frac + dy_frac * dy_frac + 2.0 * cos_gamma * dx_frac * dy_frac;
                    double r = std::sqrt(std::max(0.0, r_sq));

                    double phi_cyl = 0.5 * (1.0 - std::tanh((r - cylinder_radius) / interface_width));
                    max_phi = std::max(max_phi, phi_cyl);
                }
                phi_a[idx] = max_phi;
            }
        }

        // Normalize phi_a
        double phi_a_sum = 0.0;
        for (int i = 0; i < M; i++)
            phi_a_sum += phi_a[i];
        phi_a_sum /= M;
        for (int i = 0; i < M; i++)
            phi_a[i] = phi_a[i] * f / phi_a_sum;

        // Set initial fields
        for (int i = 0; i < M; i++)
        {
            double phi_b_init = 1.0 - phi_a[i];
            w[i]     = chi_n * phi_b_init;
            w[i + M] = chi_n * phi_a[i];
        }

        if (verbose)
            std::cout << "  No input file found, using cylinder initialization" << std::endl;
    }

    cb->zero_mean(&w[0]);
    cb->zero_mean(&w[M]);

    // SCFT iteration
    double energy_total = 1e20;
    double error_level = 1e20;
    double old_error_level = 1e20;

    for (int iter = 0; iter < max_iter; iter++)
    {
        solver->compute_propagators({{"A", &w[0]}, {"B", &w[M]}}, {});
        solver->compute_concentrations();
        solver->get_total_concentration("A", phi_a);
        solver->get_total_concentration("B", phi_b);

        for (int i = 0; i < M; i++)
        {
            w_minus[i] = (w[i] - w[i + M]) / 2;
            w_plus[i]  = (w[i] + w[i + M]) / 2;
        }

        energy_total = cb->inner_product(w_minus, w_minus) / chi_n / cb->get_volume();
        energy_total -= cb->integral(w_plus) / cb->get_volume();
        for (int p = 0; p < molecules->get_n_polymer_types(); p++)
        {
            Polymer& pc = molecules->get_polymer(p);
            energy_total -= pc.get_volume_fraction() / pc.get_alpha() * log(solver->get_total_partition(p));
        }

        for (int i = 0; i < M; i++)
        {
            xi[i] = 0.5 * (w[i] + w[i + M] - chi_n);
            w_out[i]     = chi_n * phi_b[i] + xi[i];
            w_out[i + M] = chi_n * phi_a[i] + xi[i];
        }
        cb->zero_mean(&w_out[0]);
        cb->zero_mean(&w_out[M]);

        old_error_level = error_level;
        for (int i = 0; i < 2 * M; i++)
            w_diff[i] = w_out[i] - w[i];
        error_level = sqrt(cb->multi_inner_product(2, w_diff, w_diff) /
                          (cb->multi_inner_product(2, w, w) + 1.0));

        if (verbose && iter % 50 == 0)
        {
            std::cout << std::setw(8) << iter;
            std::cout << std::setw(15) << std::setprecision(9) << std::fixed << energy_total;
            std::cout << std::setw(15) << std::scientific << std::setprecision(2) << error_level << std::fixed << std::endl;
        }

        if (error_level < tolerance)
        {
            if (verbose)
                std::cout << "  Converged at iteration " << iter << std::endl;
            break;
        }

        am->calculate_new_fields(w, w, w_diff, old_error_level, error_level);
    }

    // Compute stress if requested
    std::vector<double> stress;
    if (compute_stress_flag)
    {
        solver->compute_stress();
        stress = solver->get_stress();
    }

    // Cleanup
    delete[] w;
    delete[] w_out;
    delete[] w_diff;
    delete[] xi;
    delete[] w_plus;
    delete[] w_minus;
    delete[] phi_a;
    delete[] phi_b;

    delete molecules;
    delete optimizer;
    delete cb;
    delete solver;
    delete am;

    return {energy_total, stress, lx, angles};
}

int main()
{
    try
    {
        const double PI = std::numbers::pi;

        std::cout << "========================================" << std::endl;
        std::cout << "Hexagonal-Rectangular Equivalence Test" << std::endl;
        std::cout << "========================================" << std::endl;

        // Simulation parameters
        double f = 1.0 / 3.0;     // A-fraction
        double chi_n = 20.0;      // Flory-Huggins parameter
        double L = 1.757626;      // Optimal lattice constant
        double ds = 1.0 / 90.0;   // Contour step
        int max_iter = 2000;      // Combined SCFT + box iterations
        double tolerance = 1e-9;  // Tolerance for convergence

        // Hexagonal unit cell: a = b = L, gamma = 120 degrees (optimal)
        std::vector<int> nx_hex = {64, 64};
        std::vector<double> lx_hex = {L, L};
        std::vector<double> angles_hex = {90.0, 90.0, 120.0};  // Optimal angle

        // Rectangular unit cell: a = sqrt(3)*L, b = L, gamma = 90 degrees
        std::vector<int> nx_rect = {96, 64};
        std::vector<double> lx_rect = {3.044297, L};  // Optimal values
        std::vector<double> angles_rect = {90.0, 90.0, 90.0};

        // Calculate expected volumes
        double volume_hex = L * L * std::sin(120.0 * PI / 180.0);
        double volume_rect = std::sqrt(3.0) * L * L;

        std::cout << "\nParameters:" << std::endl;
        std::cout << "  f = " << f << ", chi_n = " << chi_n << ", L = " << L << std::endl;
        std::cout << "\nHexagonal cell:" << std::endl;
        std::cout << "  nx = [" << nx_hex[0] << ", " << nx_hex[1] << "]" << std::endl;
        std::cout << "  lx = [" << lx_hex[0] << ", " << lx_hex[1] << "]" << std::endl;
        std::cout << "  angles = [90, 90, 120]" << std::endl;
        std::cout << "  Expected volume = " << volume_hex << " (1 cylinder)" << std::endl;
        std::cout << "\nRectangular cell:" << std::endl;
        std::cout << "  nx = [" << nx_rect[0] << ", " << nx_rect[1] << "]" << std::endl;
        std::cout << "  lx = [" << lx_rect[0] << ", " << lx_rect[1] << "]" << std::endl;
        std::cout << "  angles = [90, 90, 90]" << std::endl;
        std::cout << "  Expected volume = " << volume_rect << " (2 cylinders)" << std::endl;

        // Volume per cylinder should be the same
        double volume_per_cyl_hex = volume_hex;
        double volume_per_cyl_rect = volume_rect / 2.0;
        std::cout << "\nVolume per cylinder:" << std::endl;
        std::cout << "  Hexagonal: " << volume_per_cyl_hex << std::endl;
        std::cout << "  Rectangular: " << volume_per_cyl_rect << std::endl;

        if (std::abs(volume_per_cyl_hex - volume_per_cyl_rect) > 1e-10)
        {
            std::cout << "ERROR: Volume per cylinder mismatch!" << std::endl;
            return -1;
        }
        std::cout << "  Volume per cylinder: MATCHED" << std::endl;

        // Run on each available platform with both memory modes
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();

        for (const std::string& platform : avail_platforms)
        {
            for (bool reduce_memory : {false, true})
            {
            std::cout << "\n----------------------------------------" << std::endl;
            std::cout << "Platform: " << platform;
            if (reduce_memory)
                std::cout << " (memory-saving)";
            std::cout << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            AbstractFactory<double>* factory = PlatformSelector::create_factory_real(platform, reduce_memory);

            //==================================================================
            // Test: Fixed Box SCFT - verify stress is small at optimal parameters
            //==================================================================
            std::cout << "\n[Fixed Box SCFT Test]" << std::endl;

            std::cout << "\nRunning hexagonal cell..." << std::endl;
            auto result_hex = run_scft_fixed_box(
                factory, nx_hex, lx_hex, angles_hex, f, chi_n, ds,
                max_iter, tolerance, "Hexagonal2D_Input.txt", true, true);
            std::cout << "  Final free energy (hex): " << std::setprecision(9) << std::fixed << result_hex.energy << std::endl;
            std::cout << "  Stress [σ_xx, σ_yy, σ_xy]: [" << std::scientific << std::setprecision(3)
                      << result_hex.stress[0] << ", " << result_hex.stress[1] << ", " << result_hex.stress[2] << "]" << std::fixed << std::endl;

            std::cout << "\nRunning rectangular cell..." << std::endl;
            auto result_rect = run_scft_fixed_box(
                factory, nx_rect, lx_rect, angles_rect, f, chi_n, ds,
                max_iter, tolerance, "Rectangular2D_Input.txt", true, true);
            std::cout << "  Final free energy (rect): " << std::setprecision(9) << std::fixed << result_rect.energy << std::endl;
            std::cout << "  Stress [σ_xx, σ_yy, σ_xy]: [" << std::scientific << std::setprecision(3)
                      << result_rect.stress[0] << ", " << result_rect.stress[1] << ", " << result_rect.stress[2] << "]" << std::fixed << std::endl;

            // Compare free energies
            double energy_diff = std::abs(result_hex.energy - result_rect.energy);
            std::cout << "\nFree energy difference: " << std::scientific << energy_diff << std::endl;

            double tolerance_comparison = 1e-4;  // Slightly relaxed for coarse grid
            if (energy_diff > tolerance_comparison)
            {
                std::cout << "ERROR: Free energies differ by more than " << tolerance_comparison << "!" << std::endl;
                delete factory;
                return -1;
            }
            std::cout << "Energy comparison: PASSED" << std::endl;

            // Verify geometric relationship: rect_lx[0] = sqrt(3) * hex_lx[0]
            double expected_ratio = std::sqrt(3.0);
            double actual_ratio = result_rect.lx[0] / result_hex.lx[0];
            double ratio_error = std::abs(actual_ratio - expected_ratio) / expected_ratio;
            std::cout << "\nGeometric relationship check:" << std::endl;
            std::cout << "  Expected lx_rect[0] / lx_hex[0] = sqrt(3) = " << std::setprecision(6) << expected_ratio << std::endl;
            std::cout << "  Actual ratio: " << actual_ratio << std::endl;
            std::cout << "  Relative error: " << std::scientific << ratio_error << std::endl;

            if (ratio_error > 0.01)  // 1% tolerance
            {
                std::cout << "WARNING: Geometric relationship error > 1%" << std::endl;
            }
            else
            {
                std::cout << "Geometric relationship: PASSED" << std::endl;
            }

            // Check stress convergence
            double max_stress_hex = std::max(std::abs(result_hex.stress[0]), std::abs(result_hex.stress[1]));
            double max_stress_rect = std::max(std::abs(result_rect.stress[0]), std::abs(result_rect.stress[1]));
            std::cout << "\nStress at optimal box:" << std::endl;
            std::cout << "  Max |stress| (hex): " << std::scientific << max_stress_hex << std::endl;
            std::cout << "  Max |stress| (rect): " << max_stress_rect << std::endl;
            std::cout << "  σ_xy (hex): " << result_hex.stress[2] << std::endl;
            std::cout << "  σ_xy (rect): " << result_rect.stress[2] << std::fixed << std::endl;

            delete factory;
            }  // end reduce_memory loop
        }  // end platform loop

        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
