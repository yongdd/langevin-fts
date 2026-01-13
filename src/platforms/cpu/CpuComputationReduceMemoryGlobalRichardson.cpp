/**
 * @file CpuComputationReduceMemoryGlobalRichardson.cpp
 * @brief Implementation of memory-efficient Global Richardson computation.
 */

#include <cmath>
#include <cstring>
#include <omp.h>

#include "CpuComputationReduceMemoryGlobalRichardson.h"
#include "SimpsonRule.h"
#include "PropagatorCode.h"

CpuComputationReduceMemoryGlobalRichardson::CpuComputationReduceMemoryGlobalRichardson(
    ComputationBox<double>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer)
    : PropagatorComputation<double>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Global Richardson (Reduce Memory), CPU ---------" << std::endl;
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

        // Calculate checkpoint interval
        total_max_n_segment = 0;
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            total_max_n_segment += item.second.max_n_segment;
        }
        checkpoint_interval = (int)ceil(sqrt((double)total_max_n_segment));
        if (checkpoint_interval < 1)
            checkpoint_interval = 1;

        #ifndef NDEBUG
        std::cout << "Number of CPU threads: " << n_streams << std::endl;
        std::cout << "Checkpoint interval: " << checkpoint_interval << std::endl;
        #endif

        // Allocate checkpoint propagators
        if (propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("No propagator codes. Add polymers first.");

        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;

            // Number of checkpoints needed
            int n_checkpoints = max_n_segment / checkpoint_interval + 1;

            // Allocate checkpoint storage for full-step and half-step
            for (int c = 0; c < n_checkpoints; c++)
            {
                propagator_full_at_check_point[std::make_tuple(key, c)] = new double[M];
                propagator_half_at_check_point[std::make_tuple(key, c)] = new double[M];
            }

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment + 1];
            for (int i = 0; i <= max_n_segment; i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Allocate recomputation workspace
        // Size: checkpoint_interval + 1 for each
        for (int i = 0; i <= checkpoint_interval; i++)
        {
            q_full_recal.push_back(new double[M]);
            q_half_recal.push_back(new double[M]);
        }

        // Ping-pong buffers
        q_full_pair[0] = new double[M];
        q_full_pair[1] = new double[M];
        q_half_pair[0] = new double[M];
        q_half_pair[1] = new double[M];
        q_half_temp = new double[M];
        q_full_skip[0] = new double[M];
        q_full_skip[1] = new double[M];
        q_half_skip[0] = new double[M];
        q_half_skip[1] = new double[M];

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

            partition_segment_info.push_back(std::make_tuple(
                p, key_left, key_right, n_segment_left, n_aggregated));
            current_p++;
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

CpuComputationReduceMemoryGlobalRichardson::~CpuComputationReduceMemoryGlobalRichardson()
{
    delete solver;
    delete sc;

    for (const auto& item : propagator_full_at_check_point)
        delete[] item.second;

    for (const auto& item : propagator_half_at_check_point)
        delete[] item.second;

    for (auto* ptr : q_full_recal)
        delete[] ptr;

    for (auto* ptr : q_half_recal)
        delete[] ptr;

    delete[] q_full_pair[0];
    delete[] q_full_pair[1];
    delete[] q_half_pair[0];
    delete[] q_half_pair[1];
    delete[] q_half_temp;
    delete[] q_full_skip[0];
    delete[] q_full_skip[1];
    delete[] q_half_skip[0];
    delete[] q_half_skip[1];

    for (const auto& item : phi_block)
        delete[] item.second;

    for (const auto& item : phi_solvent)
        delete[] item;

    #ifndef NDEBUG
    for (const auto& item : propagator_finished)
        delete[] item.second;
    #endif
}

void CpuComputationReduceMemoryGlobalRichardson::update_laplacian_operator()
{
    solver->update_laplacian_operator();
}

void CpuComputationReduceMemoryGlobalRichardson::compute_propagators(
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
            int max_n_segment = item.second.max_n_segment;
            for (int i = 0; i <= max_n_segment; i++)
                propagator_finished[key][i] = false;
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

                // Local ping-pong buffers for this thread
                double* q_full_curr = new double[M];
                double* q_full_next = new double[M];
                double* q_half_curr = new double[M];
                double* q_half_next = new double[M];
                double* q_half_mid = new double[M];

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
                            q_full_curr[i] = q_init[g][i];
                            q_half_curr[i] = q_init[g][i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < M; i++)
                        {
                            q_full_curr[i] = 1.0;
                            q_half_curr[i] = 1.0;
                        }
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                else if (n_segment_from == 0 && deps.size() > 0)
                {
                    // Junction node - need to get from checkpoints of dependencies
                    if (key[0] == '[')
                    {
                        // Aggregated (sum)
                        for (int i = 0; i < M; i++)
                        {
                            q_full_curr[i] = 0.0;
                            q_half_curr[i] = 0.0;
                        }

                        for (size_t d = 0; d < deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment = std::get<1>(deps[d]);
                            int sub_n_repeated = std::get<2>(deps[d]);

                            // Get from checkpoint
                            int checkpoint_idx = sub_n_segment / checkpoint_interval;
                            double* sub_full = propagator_full_at_check_point[std::make_tuple(sub_dep, checkpoint_idx)];
                            double* sub_half = propagator_half_at_check_point[std::make_tuple(sub_dep, checkpoint_idx)];

                            for (int i = 0; i < M; i++)
                            {
                                q_full_curr[i] += sub_full[i] * static_cast<double>(sub_n_repeated);
                                q_half_curr[i] += sub_half[i] * static_cast<double>(sub_n_repeated);
                            }
                        }
                    }
                    else if (key[0] == '(')
                    {
                        // Product (junction)
                        for (int i = 0; i < M; i++)
                        {
                            q_full_curr[i] = 1.0;
                            q_half_curr[i] = 1.0;
                        }

                        for (size_t d = 0; d < deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment = std::get<1>(deps[d]);
                            int sub_n_repeated = std::get<2>(deps[d]);

                            // Get from checkpoint
                            int checkpoint_idx = sub_n_segment / checkpoint_interval;
                            double* sub_full = propagator_full_at_check_point[std::make_tuple(sub_dep, checkpoint_idx)];
                            double* sub_half = propagator_half_at_check_point[std::make_tuple(sub_dep, checkpoint_idx)];

                            for (int i = 0; i < M; i++)
                            {
                                q_full_curr[i] *= pow(sub_full[i], sub_n_repeated);
                                q_half_curr[i] *= pow(sub_half[i], sub_n_repeated);
                            }
                        }
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                else
                {
                    // Starting from a checkpoint
                    int checkpoint_idx = n_segment_from / checkpoint_interval;
                    double* checkpoint_full = propagator_full_at_check_point[std::make_tuple(key, checkpoint_idx)];
                    double* checkpoint_half = propagator_half_at_check_point[std::make_tuple(key, checkpoint_idx)];

                    for (int i = 0; i < M; i++)
                    {
                        q_full_curr[i] = checkpoint_full[i];
                        q_half_curr[i] = checkpoint_half[i];
                    }
                }

                // Apply mask at segment 0
                if (n_segment_from == 0 && q_mask != nullptr)
                {
                    for (int i = 0; i < M; i++)
                    {
                        q_full_curr[i] *= q_mask[i];
                        q_half_curr[i] *= q_mask[i];
                    }
                }

                // Store checkpoint at segment 0 if needed
                if (n_segment_from == 0)
                {
                    double* cp_full = propagator_full_at_check_point[std::make_tuple(key, 0)];
                    double* cp_half = propagator_half_at_check_point[std::make_tuple(key, 0)];
                    for (int i = 0; i < M; i++)
                    {
                        cp_full[i] = q_full_curr[i];
                        cp_half[i] = q_half_curr[i];
                    }
                }

                // Advance propagators
                for (int n = n_segment_from; n < n_segment_to; n++)
                {
                    // Full-step chain: one full step
                    solver->advance_full_step(q_full_curr, q_full_next, monomer_type, q_mask);

                    // Half-step chain: two half steps
                    solver->advance_half_step(q_half_curr, q_half_mid, monomer_type, q_mask);
                    solver->advance_half_step(q_half_mid, q_half_next, monomer_type, q_mask);

                    // Swap buffers
                    std::swap(q_full_curr, q_full_next);
                    std::swap(q_half_curr, q_half_next);

                    // Store checkpoint if at checkpoint position
                    if ((n + 1) % checkpoint_interval == 0 || n + 1 == n_segment_to)
                    {
                        int checkpoint_idx = (n + 1) / checkpoint_interval;
                        double* cp_full = propagator_full_at_check_point[std::make_tuple(key, checkpoint_idx)];
                        double* cp_half = propagator_half_at_check_point[std::make_tuple(key, checkpoint_idx)];
                        for (int i = 0; i < M; i++)
                        {
                            cp_full[i] = q_full_curr[i];
                            cp_half[i] = q_half_curr[i];
                        }
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][n + 1] = true;
                    #endif
                }

                delete[] q_full_curr;
                delete[] q_full_next;
                delete[] q_half_curr;
                delete[] q_half_next;
                delete[] q_half_mid;
            }
        }

        // Compute partition functions using Richardson extrapolation
        for (const auto& segment_info : partition_segment_info)
        {
            int p = std::get<0>(segment_info);
            std::string key_left = std::get<1>(segment_info);
            std::string key_right = std::get<2>(segment_info);
            int n_segment_left = std::get<3>(segment_info);
            int n_aggregated = std::get<4>(segment_info);

            // Get checkpoints for q_left at n_segment_left and q_right at 0
            int checkpoint_left = n_segment_left / checkpoint_interval;
            int checkpoint_right = 0;

            double* q_full_left = propagator_full_at_check_point[std::make_tuple(key_left, checkpoint_left)];
            double* q_half_left = propagator_half_at_check_point[std::make_tuple(key_left, checkpoint_left)];
            double* q_full_right = propagator_full_at_check_point[std::make_tuple(key_right, checkpoint_right)];
            double* q_half_right = propagator_half_at_check_point[std::make_tuple(key_right, checkpoint_right)];

            // Compute Richardson-extrapolated inner product
            double q_inner_product = 0.0;
            for (int i = 0; i < M; i++)
            {
                double q_rich_left = (4.0 * q_half_left[i] - q_full_left[i]) / 3.0;
                double q_rich_right = (4.0 * q_half_right[i] - q_full_right[i]) / 3.0;
                q_inner_product += q_rich_left * q_rich_right;
            }
            q_inner_product *= (cb->get_volume() / M);

            single_polymer_partitions[p] = q_inner_product / (n_aggregated * cb->get_volume());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationReduceMemoryGlobalRichardson::compute_concentrations()
{
    try
    {
        const int M = cb->get_total_grid();

        // Calculate segment concentrations with on-the-fly recomputation
        for (auto& block : phi_block)
        {
            const auto& key = block.first;

            int p = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            int n_repeated = propagator_computation_optimizer->get_computation_block(key).n_repeated;
            std::string monomer_type = propagator_computation_optimizer->get_computation_block(key).monomer_type;

            if (n_segment_right == 0)
            {
                for (int i = 0; i < M; i++)
                    block.second[i] = 0.0;
                continue;
            }

            // Compute φ with on-the-fly recomputation
            calculate_phi_one_block(
                block.second,
                key_left,
                key_right,
                n_segment_left,
                n_segment_right,
                monomer_type
            );

            // Normalize
            Polymer& pc = molecules->get_polymer(p);
            double norm = (molecules->get_ds() * pc.get_volume_fraction() / pc.get_alpha() * n_repeated) /
                         single_polymer_partitions[p];
            for (int i = 0; i < M; i++)
                block.second[i] *= norm;
        }

        // Solvent concentrations
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

void CpuComputationReduceMemoryGlobalRichardson::calculate_phi_one_block(
    double* phi,
    std::string key_left,
    std::string key_right,
    const int N_LEFT,
    const int N_RIGHT,
    std::string monomer_type)
{
    try
    {
        const int M = cb->get_total_grid();
        const double* q_mask = cb->get_mask();
        std::vector<double> simpson_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Initialize phi
        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

        // Process in blocks defined by checkpoints
        // For each contour point n in [0, N_RIGHT]:
        //   idx_left = N_LEFT - n (goes from N_LEFT down to N_LEFT - N_RIGHT)
        //   idx_right = n (goes from 0 to N_RIGHT)

        int n_start = 0;
        while (n_start <= N_RIGHT)
        {
            int n_end = std::min(n_start + checkpoint_interval, N_RIGHT);

            // Determine checkpoint positions for left and right propagators
            // For left: we need values at N_LEFT - n_start down to N_LEFT - n_end
            // For right: we need values at n_start up to n_end

            // Find checkpoint for right propagator (advancing from n_start to n_end)
            int checkpoint_right = n_start / checkpoint_interval;
            int right_checkpoint_pos = checkpoint_right * checkpoint_interval;

            // Find checkpoint for left propagator (we need N_LEFT - n_start to N_LEFT - n_end)
            int left_pos_start = N_LEFT - n_start;
            int left_checkpoint = left_pos_start / checkpoint_interval;
            int left_checkpoint_pos = left_checkpoint * checkpoint_interval;

            // Load and advance right propagator from checkpoint to n_start
            double* cp_full_right = propagator_full_at_check_point[std::make_tuple(key_right, checkpoint_right)];
            double* cp_half_right = propagator_half_at_check_point[std::make_tuple(key_right, checkpoint_right)];

            // Copy to working buffers
            for (int i = 0; i < M; i++)
            {
                q_full_pair[0][i] = cp_full_right[i];
                q_half_pair[0][i] = cp_half_right[i];
            }

            // Advance from right_checkpoint_pos to n_start
            int full_idx = 0;
            for (int n = right_checkpoint_pos; n < n_start; n++)
            {
                solver->advance_full_step(q_full_pair[full_idx], q_full_pair[1 - full_idx], monomer_type, q_mask);
                solver->advance_half_step(q_half_pair[full_idx], q_half_temp, monomer_type, q_mask);
                solver->advance_half_step(q_half_temp, q_half_pair[1 - full_idx], monomer_type, q_mask);
                full_idx = 1 - full_idx;
            }

            // Now compute and store right propagators for n_start to n_end
            for (int i = 0; i < M; i++)
            {
                q_full_recal[0][i] = q_full_pair[full_idx][i];
                q_half_recal[0][i] = q_half_pair[full_idx][i];
            }

            for (int n = n_start; n < n_end; n++)
            {
                int local_idx = n - n_start;
                solver->advance_full_step(q_full_recal[local_idx], q_full_recal[local_idx + 1], monomer_type, q_mask);
                solver->advance_half_step(q_half_recal[local_idx], q_half_temp, monomer_type, q_mask);
                solver->advance_half_step(q_half_temp, q_half_recal[local_idx + 1], monomer_type, q_mask);
            }

            // Now handle left propagator - we need it at positions N_LEFT - n for n in [n_start, n_end]
            // This means positions [N_LEFT - n_end, N_LEFT - n_start]
            // We go backwards through these

            // Load checkpoint for left propagator
            double* cp_full_left = propagator_full_at_check_point[std::make_tuple(key_left, left_checkpoint)];
            double* cp_half_left = propagator_half_at_check_point[std::make_tuple(key_left, left_checkpoint)];

            for (int i = 0; i < M; i++)
            {
                q_full_skip[0][i] = cp_full_left[i];
                q_half_skip[0][i] = cp_half_left[i];
            }

            // Advance from left_checkpoint_pos to left_pos_start = N_LEFT - n_start
            int skip_idx = 0;
            for (int n = left_checkpoint_pos; n < left_pos_start; n++)
            {
                solver->advance_full_step(q_full_skip[skip_idx], q_full_skip[1 - skip_idx], monomer_type, q_mask);
                solver->advance_half_step(q_half_skip[skip_idx], q_half_temp, monomer_type, q_mask);
                solver->advance_half_step(q_half_temp, q_half_skip[1 - skip_idx], monomer_type, q_mask);
                skip_idx = 1 - skip_idx;
            }

            // Now accumulate phi contributions
            // At n=n_start: left at N_LEFT - n_start, right at n_start
            for (int n = n_start; n <= n_end; n++)
            {
                int local_right = n - n_start;

                // Get Richardson-extrapolated propagators
                double* q_full_r = q_full_recal[local_right];
                double* q_half_r = q_half_recal[local_right];
                double* q_full_l = q_full_skip[skip_idx];
                double* q_half_l = q_half_skip[skip_idx];

                // Accumulate with Richardson extrapolation
                for (int i = 0; i < M; i++)
                {
                    double q_rich_left = (4.0 * q_half_l[i] - q_full_l[i]) / 3.0;
                    double q_rich_right = (4.0 * q_half_r[i] - q_full_r[i]) / 3.0;
                    phi[i] += simpson_coeff[n] * q_rich_left * q_rich_right;
                }

                // Advance left propagator backward (actually forward from lower position)
                // We need to go from N_LEFT - n to N_LEFT - (n+1) = N_LEFT - n - 1
                // But we're storing in forward order, so we need to recompute
                if (n < n_end)
                {
                    // Move to next position (one step backward in contour)
                    solver->advance_full_step(q_full_skip[skip_idx], q_full_skip[1 - skip_idx], monomer_type, q_mask);
                    solver->advance_half_step(q_half_skip[skip_idx], q_half_temp, monomer_type, q_mask);
                    solver->advance_half_step(q_half_temp, q_half_skip[1 - skip_idx], monomer_type, q_mask);
                    skip_idx = 1 - skip_idx;
                }
            }

            n_start = n_end + 1;
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationReduceMemoryGlobalRichardson::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    compute_propagators(w_input, q_init);
    compute_concentrations();
}

void CpuComputationReduceMemoryGlobalRichardson::compute_stress()
{
    throw_with_line_number("Stress computation not yet implemented for Global Richardson.");
}

void CpuComputationReduceMemoryGlobalRichardson::get_chain_propagator(
    double* q_out, int polymer, int v, int u, int n)
{
    throw_with_line_number("get_chain_propagator requires recomputation - not yet implemented for reduce memory.");
}

void CpuComputationReduceMemoryGlobalRichardson::get_total_concentration(
    int p, std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P - 1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P - 1) + "]");

        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

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

void CpuComputationReduceMemoryGlobalRichardson::get_total_concentration_gce(
    double fugacity, int p, std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P - 1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P - 1) + "]");

        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

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

void CpuComputationReduceMemoryGlobalRichardson::get_total_concentration(
    std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();

        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

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

void CpuComputationReduceMemoryGlobalRichardson::get_solvent_concentration(int s, double* phi)
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

bool CpuComputationReduceMemoryGlobalRichardson::check_total_partition()
{
    int n_polymer_types = molecules->get_n_polymer_types();

    std::cout << "Global Richardson (Reduce Memory) Partition Functions:" << std::endl;
    std::cout << "Polymer\tQ_richardson" << std::endl;

    for (int p = 0; p < n_polymer_types; p++)
    {
        std::cout << p << "\t" << single_polymer_partitions[p] << std::endl;
    }

    return true;
}
