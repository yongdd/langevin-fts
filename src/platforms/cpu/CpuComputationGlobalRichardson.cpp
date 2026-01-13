/**
 * @file CpuComputationGlobalRichardson.cpp
 * @brief Implementation of Global Richardson computation at quadrature level.
 *
 * Maintains two independent propagator chains and applies Richardson
 * extrapolation only when computing the partition function Q.
 */

#include <cmath>
#include <cstring>
#include <omp.h>

#include "CpuComputationGlobalRichardson.h"
#include "SimpsonRule.h"
#include "PropagatorCode.h"

CpuComputationGlobalRichardson::CpuComputationGlobalRichardson(
    ComputationBox<double>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer)
    : PropagatorComputation<double>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Global Richardson (Quadrature Level), CPU ---------" << std::endl;
        #endif

        if (molecules->get_model_name() != "continuous")
            throw_with_line_number("Global Richardson only supports 'continuous' chain model.");

        const int M = cb->get_total_grid();

        // Create base solver
        solver = new CpuSolverGlobalRichardsonBase(cb, molecules);

        // Set up parallel threads
        const char* ENV_OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
        std::string env_omp_num_threads(ENV_OMP_NUM_THREADS ? ENV_OMP_NUM_THREADS : "");
        if (env_omp_num_threads.empty())
            n_streams = 8;
        else
            n_streams = std::stoi(env_omp_num_threads);

        #ifndef NDEBUG
        std::cout << "Number of CPU threads: " << n_streams << std::endl;
        #endif

        // Allocate propagator arrays
        if (propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("No propagator codes. Add polymers first.");

        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;

            // Full-step: N+1 propagators
            int full_size = max_n_segment + 1;
            propagator_full_size[key] = full_size;
            propagator_full[key] = new double*[full_size];
            for (int i = 0; i < full_size; i++)
                propagator_full[key][i] = new double[M];

            // Half-step: 2N+1 propagators
            int half_size = 2 * max_n_segment + 1;
            propagator_half_size[key] = half_size;
            propagator_half[key] = new double*[half_size];
            for (int i = 0; i < half_size; i++)
                propagator_half[key][i] = new double[M];

            // Richardson-extrapolated: N+1 propagators (same size as full)
            propagator_richardson_size[key] = full_size;
            propagator_richardson[key] = new double*[full_size];
            for (int i = 0; i < full_size; i++)
                propagator_richardson[key][i] = new double[M];

            #ifndef NDEBUG
            propagator_full_finished[key] = new bool[full_size];
            propagator_half_finished[key] = new bool[half_size];
            for (int i = 0; i < full_size; i++)
                propagator_full_finished[key][i] = false;
            for (int i = 0; i < half_size; i++)
                propagator_half_finished[key][i] = false;
            #endif
        }

        // Allocate concentration arrays
        if (propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("No blocks. Add polymers first.");

        for (const auto& item : propagator_computation_optimizer->get_computation_blocks())
        {
            phi_block[item.first] = new double[M]();
        }

        // Set up partition function calculation info
        int current_p = 0;
        for (const auto& block : phi_block)
        {
            const auto& key = block.first;
            int p = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            if (p != current_p)
                continue;

            int n_aggregated = propagator_computation_optimizer->get_computation_block(key).v_u.size() /
                              propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;

            // Store info for Richardson propagators (computed after advancing chains)
            partition_segment_info.push_back(std::make_tuple(
                p,
                propagator_richardson[key_left][n_segment_left],  // Richardson: q at end
                propagator_richardson[key_right][0],               // Richardson: q_dagger at start
                n_aggregated
            ));
            current_p++;
        }

        // Initialize partition function storage (base class vectors already sized)
        int n_polymer_types = molecules->get_n_polymer_types();
        for (int p = 0; p < n_polymer_types; p++)
        {
            single_polymer_partitions_full[p] = 0.0;
            single_polymer_partitions_half[p] = 0.0;
        }

        // Solvent concentrations
        for (int s = 0; s < molecules->get_n_solvent_types(); s++)
        {
            phi_solvent.push_back(new double[M]);
        }

        // Create scheduler
        sc = new Scheduler(propagator_computation_optimizer->get_computation_propagators(), n_streams);

        solver->update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

CpuComputationGlobalRichardson::~CpuComputationGlobalRichardson()
{
    delete solver;
    delete sc;

    for (const auto& item : propagator_full)
    {
        for (int i = 0; i < propagator_full_size[item.first]; i++)
            delete[] item.second[i];
        delete[] item.second;
    }

    for (const auto& item : propagator_half)
    {
        for (int i = 0; i < propagator_half_size[item.first]; i++)
            delete[] item.second[i];
        delete[] item.second;
    }

    for (const auto& item : propagator_richardson)
    {
        for (int i = 0; i < propagator_richardson_size[item.first]; i++)
            delete[] item.second[i];
        delete[] item.second;
    }

    for (const auto& item : phi_block)
        delete[] item.second;

    for (const auto& item : phi_solvent)
        delete[] item;

    #ifndef NDEBUG
    for (const auto& item : propagator_full_finished)
        delete[] item.second;
    for (const auto& item : propagator_half_finished)
        delete[] item.second;
    #endif
}

void CpuComputationGlobalRichardson::update_laplacian_operator()
{
    solver->update_laplacian_operator();
}

void CpuComputationGlobalRichardson::compute_propagators(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    try
    {
        const int M = cb->get_total_grid();

        // Validate inputs
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            if (!w_input.contains(item.second.monomer_type))
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" not in w_input.");
        }

        // Update Boltzmann factors
        solver->update_dw(w_input);

        const double* q_mask = cb->get_mask();

        #ifndef NDEBUG
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            for (int i = 0; i < propagator_full_size[key]; i++)
                propagator_full_finished[key][i] = false;
            for (int i = 0; i < propagator_half_size[key]; i++)
                propagator_half_finished[key][i] = false;
        }
        #endif

        // Process propagators according to schedule
        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            #pragma omp parallel for num_threads(n_streams)
            for (size_t job = 0; job < parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                double** _prop_full = propagator_full[key];
                double** _prop_half = propagator_half[key];

                // Initialize at segment 0
                if (n_segment_from == 0 && deps.size() == 0)
                {
                    // Leaf node
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (!q_init.contains(g))
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        for (int i = 0; i < M; i++)
                        {
                            _prop_full[0][i] = q_init[g][i];
                            _prop_half[0][i] = q_init[g][i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < M; i++)
                        {
                            _prop_full[0][i] = 1.0;
                            _prop_half[0][i] = 1.0;
                        }
                    }

                    #ifndef NDEBUG
                    propagator_full_finished[key][0] = true;
                    propagator_half_finished[key][0] = true;
                    #endif
                }
                else if (n_segment_from == 0 && deps.size() > 0)
                {
                    // Junction node
                    if (key[0] == '[')
                    {
                        // Aggregated (sum)
                        for (int i = 0; i < M; i++)
                        {
                            _prop_full[0][i] = 0.0;
                            _prop_half[0][i] = 0.0;
                        }

                        for (size_t d = 0; d < deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment = std::get<1>(deps[d]);
                            int sub_n_repeated = std::get<2>(deps[d]);

                            double** _sub_full = propagator_full[sub_dep];
                            double** _sub_half = propagator_half[sub_dep];

                            for (int i = 0; i < M; i++)
                            {
                                _prop_full[0][i] += _sub_full[sub_n_segment][i] * static_cast<double>(sub_n_repeated);
                                _prop_half[0][i] += _sub_half[2 * sub_n_segment][i] * static_cast<double>(sub_n_repeated);
                            }
                        }
                    }
                    else if (key[0] == '(')
                    {
                        // Product (junction)
                        for (int i = 0; i < M; i++)
                        {
                            _prop_full[0][i] = 1.0;
                            _prop_half[0][i] = 1.0;
                        }

                        for (size_t d = 0; d < deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment = std::get<1>(deps[d]);
                            int sub_n_repeated = std::get<2>(deps[d]);

                            double** _sub_full = propagator_full[sub_dep];
                            double** _sub_half = propagator_half[sub_dep];

                            for (int i = 0; i < M; i++)
                            {
                                _prop_full[0][i] *= pow(_sub_full[sub_n_segment][i], sub_n_repeated);
                                _prop_half[0][i] *= pow(_sub_half[2 * sub_n_segment][i], sub_n_repeated);
                            }
                        }
                    }

                    #ifndef NDEBUG
                    propagator_full_finished[key][0] = true;
                    propagator_half_finished[key][0] = true;
                    #endif
                }

                // Apply mask at segment 0
                if (n_segment_from == 0 && q_mask != nullptr)
                {
                    for (int i = 0; i < M; i++)
                    {
                        _prop_full[0][i] *= q_mask[i];
                        _prop_half[0][i] *= q_mask[i];
                    }
                }

                // Advance propagators
                for (int n = n_segment_from; n < n_segment_to; n++)
                {
                    // Full-step chain: one full step
                    solver->advance_full_step(
                        _prop_full[n], _prop_full[n + 1], monomer_type, q_mask);

                    // Half-step chain: two half steps
                    // n=0 -> 0,1,2; n=1 -> 2,3,4; etc.
                    int half_idx = 2 * n;
                    solver->advance_half_step(
                        _prop_half[half_idx], _prop_half[half_idx + 1], monomer_type, q_mask);
                    solver->advance_half_step(
                        _prop_half[half_idx + 1], _prop_half[half_idx + 2], monomer_type, q_mask);

                    #ifndef NDEBUG
                    propagator_full_finished[key][n + 1] = true;
                    propagator_half_finished[key][half_idx + 1] = true;
                    propagator_half_finished[key][half_idx + 2] = true;
                    #endif
                }
            }
        }

        // Compute Richardson-extrapolated propagators: q_rich[n] = (4·q_half[2n] - q_full[n]) / 3
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int n_segment = propagator_richardson_size[key] - 1;

            double** _prop_full = propagator_full[key];
            double** _prop_half = propagator_half[key];
            double** _prop_rich = propagator_richardson[key];

            #pragma omp parallel for num_threads(n_streams)
            for (int n = 0; n <= n_segment; n++)
            {
                for (int i = 0; i < M; i++)
                {
                    _prop_rich[n][i] = (4.0 * _prop_half[2 * n][i] - _prop_full[n][i]) / 3.0;
                }
            }
        }

        // Compute partition functions using Richardson propagators
        for (const auto& segment_info : partition_segment_info)
        {
            int p = std::get<0>(segment_info);
            double* q_rich_left = std::get<1>(segment_info);
            double* q_rich_right = std::get<2>(segment_info);
            int n_aggregated = std::get<3>(segment_info);

            // Partition function from Richardson propagators
            single_polymer_partitions[p] = cb->inner_product(q_rich_left, q_rich_right) /
                                          (n_aggregated * cb->get_volume());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationGlobalRichardson::compute_concentrations()
{
    try
    {
        const int M = cb->get_total_grid();

        // Calculate segment concentrations using Richardson-extrapolated propagators
        // φ = ds * Σ simpson_coeff[n] * q_rich_1[N_LEFT-n] * q_rich_2[n]
        #pragma omp parallel for num_threads(n_streams)
        for (size_t b = 0; b < phi_block.size(); b++)
        {
            auto block = phi_block.begin();
            advance(block, b);
            const auto& key = block->first;

            int p = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            int n_repeated = propagator_computation_optimizer->get_computation_block(key).n_repeated;

            if (n_segment_right == 0)
            {
                for (int i = 0; i < M; i++)
                    block->second[i] = 0.0;
                continue;
            }

            // Compute φ using pre-computed Richardson propagators
            calculate_phi_one_block(
                block->second,
                propagator_richardson[key_left],
                propagator_richardson[key_right],
                n_segment_left,
                n_segment_right
            );

            // Normalize by Richardson-extrapolated Q
            Polymer& pc = molecules->get_polymer(p);
            double norm = (molecules->get_ds() * pc.get_volume_fraction() / pc.get_alpha() * n_repeated) /
                         single_polymer_partitions[p];
            for (int i = 0; i < M; i++)
                block->second[i] *= norm;
        }

        // Solvent concentrations not yet implemented
        if (molecules->get_n_solvent_types() > 0)
        {
            throw_with_line_number("Solvent concentration not yet implemented for Global Richardson.");
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationGlobalRichardson::calculate_phi_one_block(
    double* phi,
    double** q_1_richardson,
    double** q_2_richardson,
    const int N_LEFT,
    const int N_RIGHT)
{
    try
    {
        const int M = cb->get_total_grid();
        std::vector<double> simpson_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Compute φ using pre-computed Richardson-extrapolated propagators:
        // φ = Σ simpson_coeff[n] * q_rich_1[N_LEFT-n] * q_rich_2[n]

        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

        for (int n = 0; n <= N_RIGHT; n++)
        {
            int idx_1 = N_LEFT - n;
            int idx_2 = n;

            for (int i = 0; i < M; i++)
            {
                phi[i] += simpson_coeff[n] * q_1_richardson[idx_1][i] * q_2_richardson[idx_2][i];
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationGlobalRichardson::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    compute_propagators(w_input, q_init);
    compute_concentrations();
}

void CpuComputationGlobalRichardson::compute_stress()
{
    throw_with_line_number("Stress computation not yet implemented for Global Richardson.");
}

void CpuComputationGlobalRichardson::get_chain_propagator(
    double* q_out, int polymer, int v, int u, int n)
{
    try
    {
        const int M = cb->get_total_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v, u);

        if (!propagator_computation_optimizer->get_computation_propagators().contains(dep))
            throw_with_line_number("Could not find propagator code '" + dep + "'.");

        // Return the half-step propagator (more accurate)
        // Note: n is in HALF-step units (0..2N)
        const int N_HALF = propagator_half_size[dep] - 1;
        if (n < 0 || n > N_HALF)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_HALF) + "]");

        double** _prop_half = propagator_half[dep];
        for (int i = 0; i < M; i++)
            q_out[i] = _prop_half[n][i];
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationGlobalRichardson::get_total_concentration(
    int p, std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P - 1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P - 1) + "]");

        // Initialize array
        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

        // For each block
        for (const auto& block : phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                for (int i = 0; i < M; i++)
                    phi[i] += block.second[i];
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationGlobalRichardson::get_total_concentration_gce(
    double fugacity, int p, std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P - 1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P - 1) + "]");

        // Initialize array
        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

        // For each block
        for (const auto& block : phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                Polymer& pc = molecules->get_polymer(p);
                double norm = fugacity / pc.get_volume_fraction() * pc.get_alpha() * single_polymer_partitions[p];
                for (int i = 0; i < M; i++)
                    phi[i] += block.second[i] * norm;
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationGlobalRichardson::get_total_concentration(
    std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();

        // Zero out phi
        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

        // Sum contributions from all blocks with matching monomer type
        for (const auto& block : phi_block)
        {
            const auto& key = block.first;
            std::string block_monomer_type = propagator_computation_optimizer->get_computation_block(key).monomer_type;

            if (block_monomer_type == monomer_type)
            {
                for (int i = 0; i < M; i++)
                    phi[i] += block.second[i];
            }
        }

        // Add solvent contribution if any
        for (int s = 0; s < molecules->get_n_solvent_types(); s++)
        {
            std::string solvent_monomer_type = std::get<1>(molecules->get_solvent(s));
            if (solvent_monomer_type == monomer_type)
            {
                for (int i = 0; i < M; i++)
                    phi[i] += phi_solvent[s][i];
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationGlobalRichardson::get_solvent_concentration(int s, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();

        if (s < 0 || s >= molecules->get_n_solvent_types())
            throw_with_line_number("Invalid solvent index: " + std::to_string(s));

        for (int i = 0; i < M; i++)
            phi[i] = phi_solvent[s][i];
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

bool CpuComputationGlobalRichardson::check_total_partition()
{
    int n_polymer_types = molecules->get_n_polymer_types();

    std::cout << "Global Richardson Partition Functions:" << std::endl;
    std::cout << "Polymer\tQ_full\t\tQ_half\t\tQ_richardson" << std::endl;

    for (int p = 0; p < n_polymer_types; p++)
    {
        std::cout << p << "\t"
                  << single_polymer_partitions_full[p] << "\t"
                  << single_polymer_partitions_half[p] << "\t"
                  << single_polymer_partitions[p] << std::endl;
    }

    // Check consistency across blocks (using half-step propagators)
    std::vector<std::vector<double>> total_partitions;
    for (int p = 0; p < n_polymer_types; p++)
        total_partitions.push_back(std::vector<double>());

    for (const auto& block : phi_block)
    {
        const auto& key = block.first;
        int p = std::get<0>(key);
        std::string key_left = std::get<1>(key);
        std::string key_right = std::get<2>(key);

        int n_segment_right = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated = propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators = propagator_computation_optimizer->get_computation_block(key).v_u.size();

        // Check at half-step points
        for (int n = 0; n <= 2 * n_segment_right; n++)
        {
            double total_partition = cb->inner_product(
                propagator_half[key_left][2 * n_segment_left - n],
                propagator_half[key_right][n]) * (n_repeated / cb->get_volume());

            total_partition /= n_propagators;
            total_partitions[p].push_back(total_partition);
        }
    }

    // Check min/max difference
    std::cout << "Polymer id: max, min, diff of total partitions (half-step)" << std::endl;
    for (size_t p = 0; p < total_partitions.size(); p++)
    {
        double max_partition = -1e20;
        double min_partition = 1e20;
        for (size_t n = 0; n < total_partitions[p].size(); n++)
        {
            if (std::abs(total_partitions[p][n]) > max_partition)
                max_partition = std::abs(total_partitions[p][n]);
            if (std::abs(total_partitions[p][n]) < min_partition)
                min_partition = std::abs(total_partitions[p][n]);
        }
        double diff = std::abs(max_partition - min_partition);

        std::cout << "\t" << p << ": " << max_partition << ", " << min_partition << ", " << diff << std::endl;

        if (diff > 1e-7)  // Tolerance for half-step chain
            return false;
    }

    return true;
}
