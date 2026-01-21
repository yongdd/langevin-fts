/**
 * @file benchmark_sdc_64cubed.cpp
 * @brief Benchmark SDC solver with 64^3 grid to measure warm-start benefit.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"

#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include "CudaComputationContinuous.h"
#endif

int main()
{
#ifdef USE_CUDA
    try
    {
        std::cout << "============================================================" << std::endl;
        std::cout << "SDC Benchmark: 64^3 Grid (262144 points)" << std::endl;
        std::cout << "============================================================" << std::endl;

        // Setup
        std::vector<int> nx = {64, 64, 64};
        std::vector<double> lx = {8.0, 8.0, 8.0};
        std::vector<std::string> bc = {"periodic", "periodic", "periodic"};
        double ds = 1.0 / 64;

        int n_grid = nx[0] * nx[1] * nx[2];

        // Create molecules
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks = {
            {"A", 0.5, 0, 1},
            {"B", 0.5, 1, 2}
        };

        Molecules* molecules = new Molecules("Continuous", ds, bond_lengths);
        molecules->add_polymer(1.0, blocks, {});

        // Create optimizer
        PropagatorComputationOptimizer prop_opt(molecules, false);

        // Generate random fields
        std::vector<double> w_A(n_grid), w_B(n_grid);
        std::mt19937 gen(42);
        std::normal_distribution<> dist(0.0, 5.0);
        for (int i = 0; i < n_grid; ++i)
        {
            w_A[i] = dist(gen);
            w_B[i] = -w_A[i];
        }

        // Create SDC solver
        CudaComputationBox<double>* cb = new CudaComputationBox<double>(nx, lx, bc);
        CudaComputationContinuous<double>* solver = new CudaComputationContinuous<double>(
            cb, molecules, &prop_opt, "realspace", "sdc-2");

        // Warm-up run
        solver->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});

        // Benchmark runs
        const int n_runs = 10;
        std::vector<double> times;

        std::cout << std::endl;
        std::cout << "Running " << n_runs << " iterations..." << std::endl;

        for (int i = 0; i < n_runs; ++i)
        {
            // Slightly perturb fields
            for (int j = 0; j < n_grid; ++j)
            {
                w_A[j] += 0.001 * (dist(gen) - 0.5);
                w_B[j] = -w_A[j];
            }

            auto start = std::chrono::high_resolution_clock::now();
            solver->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(elapsed);
            std::cout << "  Run " << (i + 1) << ": " << std::fixed << std::setprecision(2) << elapsed << " ms" << std::endl;
        }

        // Compute statistics
        double sum = 0.0;
        for (double t : times) sum += t;
        double mean = sum / times.size();

        double var = 0.0;
        for (double t : times) var += (t - mean) * (t - mean);
        double std_dev = std::sqrt(var / times.size());

        std::cout << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Results:" << std::endl;
        std::cout << "  Grid: " << nx[0] << "x" << nx[1] << "x" << nx[2] << " = " << n_grid << " points" << std::endl;
        std::cout << "  ds: " << ds << " (N_s = " << int(1.0/ds) << " contour steps)" << std::endl;
        std::cout << "  Mean time: " << std::fixed << std::setprecision(2) << mean << " ms" << std::endl;
        std::cout << "  Std dev:   " << std::fixed << std::setprecision(2) << std_dev << " ms" << std::endl;
        std::cout << "============================================================" << std::endl;

        delete solver;
        delete cb;
        delete molecules;

        return 0;
    }
    catch (std::exception& exc)
    {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
#else
    std::cerr << "CUDA not available" << std::endl;
    return 1;
#endif
}
