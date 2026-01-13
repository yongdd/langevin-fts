/**
 * @file CudaComputationGlobalRichardson.cu
 * @brief GPU propagator computation with Global Richardson extrapolation.
 *
 * Maintains two independent propagator chains on GPU and applies Richardson
 * extrapolation only when computing the partition function Q.
 */

#include <cmath>
#include <omp.h>

#include "CudaComputationGlobalRichardson.h"
#include "CudaComputationBox.h"
#include "SimpsonRule.h"
#include "PropagatorCode.h"

CudaComputationGlobalRichardson::CudaComputationGlobalRichardson(
    ComputationBox<double>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer)
    : PropagatorComputation<double>(cb, molecules, propagator_computation_optimizer),
      cb(cb), molecules(molecules),
      propagator_computation_optimizer(propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Global Richardson (Quadrature Level), CUDA ---------" << std::endl;
        #endif

        if (molecules->get_model_name() != "continuous")
            throw_with_line_number("Global Richardson only supports 'continuous' chain model.");

        const int M = cb->get_total_grid();

        // Set up parallel streams
        const char* ENV_OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
        std::string env_omp_num_threads(ENV_OMP_NUM_THREADS ? ENV_OMP_NUM_THREADS : "");
        if (env_omp_num_threads.empty())
            n_streams = MAX_STREAMS;
        else
            n_streams = std::min(std::stoi(env_omp_num_threads), MAX_STREAMS);

        #ifndef NDEBUG
        std::cout << "Number of CUDA streams: " << n_streams << std::endl;
        #endif

        // Create CUDA streams
        for (int i = 0; i < n_streams; i++)
        {
            gpu_error_check(cudaStreamCreate(&streams[i][0]));  // kernel
            gpu_error_check(cudaStreamCreate(&streams[i][1]));  // memcpy
        }

        // Create base solver
        solver = new CudaSolverGlobalRichardsonBase(cb, molecules, n_streams, streams);

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
            d_propagator_full[key] = new double*[full_size];
            for (int i = 0; i < full_size; i++)
                gpu_error_check(cudaMalloc((void**)&d_propagator_full[key][i], sizeof(double)*M));

            // Half-step: N+1 propagators (only store even positions 0, 2, 4, ..., 2N)
            // Odd positions (1, 3, 5, ...) are computed on-the-fly but not stored
            int half_size = max_n_segment + 1;
            propagator_half_size[key] = half_size;
            d_propagator_half[key] = new double*[half_size];
            for (int i = 0; i < half_size; i++)
                gpu_error_check(cudaMalloc((void**)&d_propagator_half[key][i], sizeof(double)*M));

            // Richardson-extrapolated: N+1 propagators (same size as full)
            d_propagator_richardson[key] = new double*[full_size];
            for (int i = 0; i < full_size; i++)
                gpu_error_check(cudaMalloc((void**)&d_propagator_richardson[key][i], sizeof(double)*M));

            #ifndef NDEBUG
            propagator_finished[key] = new bool[full_size];
            for (int i = 0; i < full_size; i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Allocate concentration arrays
        if (propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("No blocks. Add polymers first.");

        for (const auto& item : propagator_computation_optimizer->get_computation_blocks())
        {
            gpu_error_check(cudaMalloc((void**)&d_phi_block[item.first], sizeof(double)*M));
            gpu_error_check(cudaMemset(d_phi_block[item.first], 0, sizeof(double)*M));
        }

        // Set up partition function calculation info
        int current_p = 0;
        for (const auto& block : d_phi_block)
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

            // Store info for Richardson propagators
            partition_segment_info.push_back(std::make_tuple(
                p,
                d_propagator_richardson[key_left][n_segment_left],
                d_propagator_richardson[key_right][0],
                n_aggregated
            ));
            current_p++;
        }

        // Solvent concentrations
        for (int s = 0; s < molecules->get_n_solvent_types(); s++)
        {
            double* d_phi_s;
            gpu_error_check(cudaMalloc((void**)&d_phi_s, sizeof(double)*M));
            d_phi_solvent.push_back(d_phi_s);
        }

        // Create scheduler
        sc = new Scheduler(propagator_computation_optimizer->get_computation_propagators(), n_streams);

        // Allocate working arrays
        gpu_error_check(cudaMalloc((void**)&d_q_unity, sizeof(double)*M));
        double* h_unity = new double[M];
        for (int i = 0; i < M; i++)
            h_unity[i] = 1.0;
        gpu_error_check(cudaMemcpy(d_q_unity, h_unity, sizeof(double)*M, cudaMemcpyHostToDevice));
        delete[] h_unity;

        // Copy mask if present
        if (cb->get_mask() != nullptr)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_mask, sizeof(double)*M));
            gpu_error_check(cudaMemcpy(d_q_mask, cb->get_mask(), sizeof(double)*M, cudaMemcpyHostToDevice));
        }
        else
            d_q_mask = nullptr;

        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M));

        // Allocate per-stream temporary buffers for half-step advancement
        for (int i = 0; i < n_streams; i++)
            gpu_error_check(cudaMalloc((void**)&d_q_half_temp[i], sizeof(double)*M));

        solver->update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

CudaComputationGlobalRichardson::~CudaComputationGlobalRichardson()
{
    delete solver;
    delete sc;

    for (const auto& item : d_propagator_full)
    {
        for (int i = 0; i < propagator_full_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }

    for (const auto& item : d_propagator_half)
    {
        for (int i = 0; i < propagator_half_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }

    for (const auto& item : d_propagator_richardson)
    {
        for (int i = 0; i < propagator_full_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }

    for (const auto& item : d_phi_block)
        cudaFree(item.second);

    for (const auto& item : d_phi_solvent)
        cudaFree(item);

    #ifndef NDEBUG
    for (const auto& item : propagator_finished)
        delete[] item.second;
    #endif

    cudaFree(d_q_unity);
    if (d_q_mask != nullptr)
        cudaFree(d_q_mask);
    cudaFree(d_phi);

    // Free per-stream temporary buffers
    for (int i = 0; i < n_streams; i++)
        cudaFree(d_q_half_temp[i]);

    // Destroy streams
    for (int i = 0; i < n_streams; i++)
    {
        cudaStreamDestroy(streams[i][0]);
        cudaStreamDestroy(streams[i][1]);
    }
}

void CudaComputationGlobalRichardson::update_laplacian_operator()
{
    solver->update_laplacian_operator();
}

void CudaComputationGlobalRichardson::compute_propagators(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();

        // Validate inputs
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            if (w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" not in w_input.");
        }

        // Update Boltzmann factors (w_input is on host)
        solver->update_dw("cpu", w_input);

        #ifndef NDEBUG
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            for (int i = 0; i < propagator_full_size[key]; i++)
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
                const int STREAM = omp_get_thread_num();

                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                double** _d_prop_full = d_propagator_full[key];
                double** _d_prop_half = d_propagator_half[key];

                // Initialize at segment 0
                if (n_segment_from == 0 && deps.size() == 0)
                {
                    // Leaf node
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpyAsync(_d_prop_full[0], q_init[g],
                            sizeof(double)*M, cudaMemcpyHostToDevice, streams[STREAM][0]));
                        gpu_error_check(cudaMemcpyAsync(_d_prop_half[0], q_init[g],
                            sizeof(double)*M, cudaMemcpyHostToDevice, streams[STREAM][0]));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpyAsync(_d_prop_full[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                        gpu_error_check(cudaMemcpyAsync(_d_prop_half[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                else if (n_segment_from == 0 && deps.size() > 0)
                {
                    // Junction node
                    if (key[0] == '[')
                    {
                        // Aggregated (sum)
                        gpu_error_check(cudaMemsetAsync(_d_prop_full[0], 0, sizeof(double)*M, streams[STREAM][0]));
                        gpu_error_check(cudaMemsetAsync(_d_prop_half[0], 0, sizeof(double)*M, streams[STREAM][0]));

                        for (size_t d = 0; d < deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment = std::get<1>(deps[d]);
                            int sub_n_repeated = std::get<2>(deps[d]);

                            double** _d_sub_full = d_propagator_full[sub_dep];
                            double** _d_sub_half = d_propagator_half[sub_dep];

                            ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                _d_prop_full[0], 1.0, _d_prop_full[0],
                                sub_n_repeated, _d_sub_full[sub_n_segment], M);

                            // Half-step index: sub_n_segment (not 2*sub_n_segment)
                            ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                _d_prop_half[0], 1.0, _d_prop_half[0],
                                sub_n_repeated, _d_sub_half[sub_n_segment], M);
                        }
                        gpu_error_check(cudaPeekAtLastError());
                    }
                    else if (key[0] == '(')
                    {
                        // Product (junction)
                        gpu_error_check(cudaMemcpyAsync(_d_prop_full[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                        gpu_error_check(cudaMemcpyAsync(_d_prop_half[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

                        for (size_t d = 0; d < deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment = std::get<1>(deps[d]);
                            int sub_n_repeated = std::get<2>(deps[d]);

                            double** _d_sub_full = d_propagator_full[sub_dep];
                            double** _d_sub_half = d_propagator_half[sub_dep];

                            for (int r = 0; r < sub_n_repeated; r++)
                            {
                                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                    _d_prop_full[0], _d_prop_full[0],
                                    _d_sub_full[sub_n_segment], 1.0, M);

                                // Half-step index: sub_n_segment (not 2*sub_n_segment)
                                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                    _d_prop_half[0], _d_prop_half[0],
                                    _d_sub_half[sub_n_segment], 1.0, M);
                            }
                        }
                        gpu_error_check(cudaPeekAtLastError());
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }

                // Apply mask at segment 0
                if (n_segment_from == 0 && d_q_mask != nullptr)
                {
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        _d_prop_full[0], _d_prop_full[0], d_q_mask, 1.0, M);
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        _d_prop_half[0], _d_prop_half[0], d_q_mask, 1.0, M);
                    gpu_error_check(cudaPeekAtLastError());
                }

                // Advance propagators using pre-allocated per-stream temp buffer
                for (int n = n_segment_from; n < n_segment_to; n++)
                {
                    // Full-step chain: one full step
                    solver->advance_full_step(
                        STREAM, _d_prop_full[n], _d_prop_full[n + 1], monomer_type, d_q_mask);

                    // Half-step chain: two half steps
                    // Use per-stream temporary buffer for intermediate value (odd position)
                    solver->advance_half_step(
                        STREAM, _d_prop_half[n], d_q_half_temp[STREAM], monomer_type, d_q_mask);
                    solver->advance_half_step(
                        STREAM, d_q_half_temp[STREAM], _d_prop_half[n + 1], monomer_type, d_q_mask);

                    #ifndef NDEBUG
                    propagator_finished[key][n + 1] = true;
                    #endif
                }

                gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
                gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
            }
            gpu_error_check(cudaDeviceSynchronize());
        }

        // Compute Richardson-extrapolated propagators: q_rich[n] = (4·q_half[2n] - q_full[n]) / 3
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int n_segment = propagator_full_size[key] - 1;

            double** _d_prop_full = d_propagator_full[key];
            double** _d_prop_half = d_propagator_half[key];
            double** _d_prop_rich = d_propagator_richardson[key];

            for (int n = 0; n <= n_segment; n++)
            {
                // q_rich[n] = (4.0 * q_half[n] - q_full[n]) / 3.0
                // Note: q_half[n] stores the value at half-step position 2n
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(
                    _d_prop_rich[n], 4.0/3.0, _d_prop_half[n],
                    -1.0/3.0, _d_prop_full[n], M);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
        gpu_error_check(cudaDeviceSynchronize());

        // Compute partition functions using Richardson propagators
        for (const auto& segment_info : partition_segment_info)
        {
            int p = std::get<0>(segment_info);
            double* d_q_rich_left = std::get<1>(segment_info);
            double* d_q_rich_right = std::get<2>(segment_info);
            int n_aggregated = std::get<3>(segment_info);

            single_polymer_partitions[p] =
                dynamic_cast<CudaComputationBox<double>*>(cb)->inner_product_device(
                    d_q_rich_left, d_q_rich_right) / (n_aggregated * cb->get_volume());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationGlobalRichardson::compute_concentrations()
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();

        // Calculate segment concentrations using Richardson-extrapolated propagators
        for (const auto& d_block : d_phi_block)
        {
            const auto& key = d_block.first;
            int p = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            int n_repeated = propagator_computation_optimizer->get_computation_block(key).n_repeated;

            if (n_segment_right == 0)
            {
                gpu_error_check(cudaMemset(d_block.second, 0, sizeof(double)*M));
                continue;
            }

            // Compute φ using pre-computed Richardson propagators
            calculate_phi_one_block(
                d_block.second,
                d_propagator_richardson[key_left],
                d_propagator_richardson[key_right],
                n_segment_left,
                n_segment_right
            );

            // Normalize by Richardson-extrapolated Q
            Polymer& pc = molecules->get_polymer(p);
            double norm = (molecules->get_ds() * pc.get_volume_fraction() / pc.get_alpha() * n_repeated) /
                         single_polymer_partitions[p];
            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_block.second, norm, d_block.second, 0.0, d_block.second, M);
            gpu_error_check(cudaPeekAtLastError());
        }
        gpu_error_check(cudaDeviceSynchronize());

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

void CudaComputationGlobalRichardson::calculate_phi_one_block(
    double* d_phi,
    double** d_q_1_richardson,
    double** d_q_2_richardson,
    const int N_LEFT,
    const int N_RIGHT)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();

        std::vector<double> simpson_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Compute φ using pre-computed Richardson-extrapolated propagators:
        // φ = Σ simpson_coeff[n] * q_rich_1[N_LEFT-n] * q_rich_2[n]

        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1_richardson[N_LEFT], d_q_2_richardson[0], simpson_coeff[0], M);
        for (int n = 1; n <= N_RIGHT; n++)
        {
            ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1_richardson[N_LEFT - n], d_q_2_richardson[n], simpson_coeff[n], M);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationGlobalRichardson::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    compute_propagators(w_input, q_init);
    compute_concentrations();
}

void CudaComputationGlobalRichardson::compute_stress()
{
    throw_with_line_number("Stress computation not yet implemented for Global Richardson.");
}

void CudaComputationGlobalRichardson::get_chain_propagator(
    double* q_out, int polymer, int v, int u, int n)
{
    try
    {
        const int M = cb->get_total_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v, u);

        if (propagator_computation_optimizer->get_computation_propagators().find(dep) ==
            propagator_computation_optimizer->get_computation_propagators().end())
            throw_with_line_number("Could not find propagator code '" + dep + "'.");

        // Return the Richardson-extrapolated propagator
        // Note: n is in full-step units (0..N)
        const int N_FULL = propagator_full_size[dep] - 1;
        if (n < 0 || n > N_FULL)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_FULL) + "]");

        gpu_error_check(cudaMemcpy(q_out, d_propagator_richardson[dep][n], sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationGlobalRichardson::get_total_concentration(
    int p, std::string monomer_type, double* phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P - 1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P - 1) + "]");

        // Use d_phi as temporary
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // For each block
        for (const auto& block : d_phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, block.second, M);
            }
        }
        gpu_error_check(cudaPeekAtLastError());

        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationGlobalRichardson::get_total_concentration_gce(
    double fugacity, int p, std::string monomer_type, double* phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P - 1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P - 1) + "]");

        // Use d_phi as temporary
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // For each block
        for (const auto& block : d_phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                Polymer& pc = molecules->get_polymer(p);
                double norm = fugacity / pc.get_volume_fraction() * pc.get_alpha() * single_polymer_partitions[p];
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, norm, block.second, M);
            }
        }
        gpu_error_check(cudaPeekAtLastError());

        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationGlobalRichardson::get_total_concentration(
    std::string monomer_type, double* phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();

        // Use d_phi as temporary
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // Sum contributions from all blocks with matching monomer type
        for (const auto& block : d_phi_block)
        {
            const auto& key = block.first;
            std::string block_monomer_type = propagator_computation_optimizer->get_computation_block(key).monomer_type;

            if (block_monomer_type == monomer_type)
            {
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, block.second, M);
            }
        }
        gpu_error_check(cudaPeekAtLastError());

        // Copy to host
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));

        // Add solvent contribution on host (if any)
        double* h_phi_solvent = new double[M];
        for (int s = 0; s < molecules->get_n_solvent_types(); s++)
        {
            std::string solvent_monomer_type = std::get<1>(molecules->get_solvent(s));
            if (solvent_monomer_type == monomer_type)
            {
                gpu_error_check(cudaMemcpy(h_phi_solvent, d_phi_solvent[s], sizeof(double)*M, cudaMemcpyDeviceToHost));
                for (int i = 0; i < M; i++)
                    phi[i] += h_phi_solvent[i];
            }
        }
        delete[] h_phi_solvent;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationGlobalRichardson::get_solvent_concentration(int s, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();

        if (s < 0 || s >= molecules->get_n_solvent_types())
            throw_with_line_number("Invalid solvent index: " + std::to_string(s));

        gpu_error_check(cudaMemcpy(phi, d_phi_solvent[s], sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

bool CudaComputationGlobalRichardson::check_total_partition()
{
    int n_polymer_types = molecules->get_n_polymer_types();

    std::cout << "Global Richardson (CUDA) Partition Functions:" << std::endl;
    std::cout << "Polymer\tQ_richardson" << std::endl;

    for (int p = 0; p < n_polymer_types; p++)
    {
        std::cout << p << "\t" << single_polymer_partitions[p] << std::endl;
    }

    // Check consistency across blocks (using Richardson-extrapolated propagators)
    std::vector<std::vector<double>> total_partitions;
    for (int p = 0; p < n_polymer_types; p++)
        total_partitions.push_back(std::vector<double>());

    for (const auto& block : d_phi_block)
    {
        const auto& key = block.first;
        int p = std::get<0>(key);
        std::string key_left = std::get<1>(key);
        std::string key_right = std::get<2>(key);

        int n_segment_right = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated = propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators = propagator_computation_optimizer->get_computation_block(key).v_u.size();

        // Check at Richardson-extrapolated points
        for (int n = 0; n <= n_segment_right; n++)
        {
            double total_partition = dynamic_cast<CudaComputationBox<double>*>(cb)->inner_product_device(
                d_propagator_richardson[key_left][n_segment_left - n],
                d_propagator_richardson[key_right][n]) * (n_repeated / cb->get_volume());

            total_partition /= n_propagators;
            total_partitions[p].push_back(total_partition);
        }
    }

    // Check min/max difference
    std::cout << "Polymer id: max, min, diff of total partitions (Richardson)" << std::endl;
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

        if (diff > 1e-7)  // Tolerance for Richardson-extrapolated chain
            return false;
    }

    return true;
}
