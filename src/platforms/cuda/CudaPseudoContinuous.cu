#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "CudaPseudoContinuous.h"
#include "CudaComputationBox.h"
#include "SimpsonQuadrature.h"

CudaPseudoContinuous::CudaPseudoContinuous(
    ComputationBox *cb,
    Mixture *mx)
    : Pseudo(cb, mx)
{
    try{
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // allocate memory for partition functions
        if( mx->get_unique_branches().size() == 0)
            throw_with_line_number("There is no unique branch. Add polymers first.");
        for(const auto& item: mx->get_unique_branches())
        {
            std::string dep = item.first;
            int max_n_segment = item.second.max_n_segment;
            d_unique_partition[dep] = new double*[max_n_segment+1];
            d_unique_partition_size[dep] = max_n_segment+1;
            for(int i=0; i<d_unique_partition_size[dep]; i++)
                gpu_error_check(cudaMalloc((void**)&d_unique_partition[dep][i], sizeof(double)*M));

            #ifndef NDEBUG
            unique_partition_finished[dep] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                unique_partition_finished[dep][i] = false;
            #endif
        }

        // allocate memory for concentrations
        if( mx->get_unique_blocks().size() == 0)
            throw_with_line_number("There is no unique block. Add polymers first.");
        for(const auto& item: mx->get_unique_blocks())
        {
            d_unique_phi[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_unique_phi[item.first], sizeof(double)*M));
        }

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: mx->get_bond_lengths()){
            std::string monomer_type = item.first;
            d_boltz_bond     [monomer_type] = nullptr;
            d_boltz_bond_half[monomer_type] = nullptr;
            d_exp_dw         [monomer_type] = nullptr;
            d_exp_dw_half    [monomer_type] = nullptr;

            gpu_error_check(cudaMalloc((void**)&d_exp_dw         [monomer_type], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw_half    [monomer_type], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond     [monomer_type], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[monomer_type], sizeof(double)*M_COMPLEX));
        }

        // total partition functions for each polymer
        single_partitions = new double[mx->get_n_polymers()];

        // create scheduler for computation of partial partition function
        sc = new Scheduler(mx->get_unique_branches(), N_STREAM); 

        // create FFT plan
        const int NRANK{cb->get_dim()};
        int n_grid[NRANK];

        if(cb->get_dim() == 3)
        {
            n_grid[0] = cb->get_nx(0);
            n_grid[1] = cb->get_nx(1);
            n_grid[2] = cb->get_nx(2);
        }
        else if(cb->get_dim() == 2)
        {
            n_grid[0] = cb->get_nx(0);
            n_grid[1] = cb->get_nx(1);
        }
        else if(cb->get_dim() == 1)
        {
            n_grid[0] = cb->get_nx(0);
        }

        cufftPlanMany(&plan_for_1, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,1);
        cufftPlanMany(&plan_bak_1, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,1);
        cufftPlanMany(&plan_for_2, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,2);
        cufftPlanMany(&plan_bak_2, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,2);

        // allocate memory for get_concentration
        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M));

        // allocate memory for pseudo-spectral: one_step()
        gpu_error_check(cudaMalloc((void**)&d_q_step1_1, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_step2_1, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_qk_in_1,  sizeof(ftsComplex)*M_COMPLEX));

        gpu_error_check(cudaMalloc((void**)&d_q_step1_2, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_q_step2_2, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_qk_in_2,  sizeof(ftsComplex)*2*M_COMPLEX));

        // allocate memory for stress calculation: compute_stress()
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_qk_1,        sizeof(ftsComplex)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_qk_2,        sizeof(ftsComplex)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q_multi,         sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_stress_sum,      sizeof(double)*M_COMPLEX));

        update_bond_function();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoContinuous::~CudaPseudoContinuous()
{
    cufftDestroy(plan_for_1);
    cufftDestroy(plan_bak_1);
    cufftDestroy(plan_for_2);
    cufftDestroy(plan_bak_2);

    delete sc;

    delete[] single_partitions;

    for(const auto& item: d_boltz_bond)
        cudaFree(item.second);
    for(const auto& item: d_boltz_bond_half)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw_half)
        cudaFree(item.second);
    for(const auto& item: d_unique_partition)
    {
        for(int i=0; i<d_unique_partition_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: d_unique_phi)
        cudaFree(item.second);

    #ifndef NDEBUG
    for(const auto& item: unique_partition_finished)
        delete[] item.second;
    #endif

    // for get_concentration
    cudaFree(d_phi);

    // for pseudo-spectral: one_step()
    cudaFree(d_q_step1_1);
    cudaFree(d_q_step2_1);
    cudaFree(d_q_step1_2);
    cudaFree(d_q_step2_2);
    cudaFree(d_qk_in_1);
    cudaFree(d_qk_in_2);

    // for stress calculation: compute_stress()
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    cudaFree(d_qk_1);
    cudaFree(d_qk_2);
    cudaFree(d_q_multi);
    cudaFree(d_stress_sum);
}

void CudaPseudoContinuous::update_bond_function()
{
    try{
        // for pseudo-spectral: one_step()
        const int M_COMPLEX = this->n_complex_grid;
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];

        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), mx->get_ds());
            get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), mx->get_ds());
        
            gpu_error_check(cudaMemcpy(d_boltz_bond[monomer_type],      boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_boltz_bond_half[monomer_type], boltz_bond_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        }

        // for stress calculation: compute_stress()
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];
        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        gpu_error_check(cudaMemcpy(d_fourier_basis_x, fourier_basis_x, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_y, fourier_basis_y, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_z, fourier_basis_z, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
void CudaPseudoContinuous::compute_statistics(
    std::map<std::string, double*> w_input,
    std::map<std::string, double*> q_init)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const double ds = mx->get_ds();

        for(const auto& item: mx->get_unique_branches())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        for(const auto& item: w_input)
        {
            if( d_exp_dw.find(item.first) == d_exp_dw.end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in d_exp_dw.");     
        }

        // if( q_init.size() > 0)
        //     throw_with_line_number("Currently, \'q_init\' is not supported.");

        // exp_dw and exp_dw_half
        double exp_dw[M];
        double exp_dw_half[M];
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
            { 
                exp_dw     [i] = exp(-w[i]*ds*0.5);
                exp_dw_half[i] = exp(-w[i]*ds*0.25);
            }
            gpu_error_check(cudaMemcpy(d_exp_dw     [monomer_type], exp_dw,      sizeof(double)*M,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_exp_dw_half[monomer_type], exp_dw_half, sizeof(double)*M,cudaMemcpyHostToDevice));
        }

        double q_uniform[M];
        for(int i=0; i<M; i++)
            q_uniform[i] = 1.0;

        auto& branch_schedule = sc->get_schedule();
        // // display all jobs
        // int time_span_count=0;
        // for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        // {
        //     std::cout << "jobs: " << time_span_count << std::endl;
        //     for(int job=0; job<parallel_job->size(); job++)
        //     {
        //         auto& key = std::get<0>((*parallel_job)[job]);
        //         int n_segment_from = std::get<1>((*parallel_job)[job]);
        //         int n_segment_to = std::get<2>((*parallel_job)[job]);
        //         std::cout << "key, n_segment_from, n_segment_to: " + key + ", " + std::to_string(n_segment_from) + ", " + std::to_string(n_segment_to) + ". " << std::endl;
        //     }
        //     time_span_count++;
        // }
        // time_span_count=0;

        // for each time span
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // for each job
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = mx->get_unique_branch(key).deps;
                auto monomer_type = mx->get_unique_branch(key).monomer_type;

                // check key
                #ifndef NDEBUG
                if (d_unique_partition.find(key) == d_unique_partition.end())
                    throw_with_line_number("Could not find key '" + key + "'. ");
                #endif

                double **_d_unique_partition = d_unique_partition[key];

                // std::cout << std::to_string(time_span_count) + "job, key, n_segment_from: " +  ", " + key + ", " << std::to_string(n_segment_from) << std::endl;

                // if it is leaf node
                if(n_segment_from == 1 && deps.size() == 0)
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = Mixture::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(_d_unique_partition[0], q_init[g],
                            sizeof(double)*M, cudaMemcpyHostToDevice));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(_d_unique_partition[0], q_uniform,
                            sizeof(double)*M, cudaMemcpyHostToDevice));
                    }

                    #ifndef NDEBUG
                    unique_partition_finished[key][0] = true;
                    #endif
                }
                // if it is not leaf node
                else if (n_segment_from == 1 && deps.size() > 0)
                {
                    // if it is superposed
                    if (key[0] == '[')
                    {
                        // initialize to zero
                        gpu_error_check(cudaMemset(_d_unique_partition[0], 0, sizeof(double)*M));

                        // add all partition functions at junctions if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // check sub key
                            #ifndef NDEBUG
                            if (d_unique_partition.find(sub_dep) == d_unique_partition.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!unique_partition_finished[sub_dep][sub_n_segment])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            lin_comb<<<N_BLOCKS, N_THREADS>>>(
                                _d_unique_partition[0], 1.0, _d_unique_partition[0],
                                sub_n_repeated, d_unique_partition[sub_dep][sub_n_segment], M);
                        }

                        #ifndef NDEBUG
                        unique_partition_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", " << std::to_string(0) << std::endl;
                    }
                    else
                    {
                        // initialize to one
                        gpu_error_check(cudaMemcpy(_d_unique_partition[0], q_uniform,
                            sizeof(double)*M, cudaMemcpyHostToDevice));

                        // multiplay all partition functions at junctions if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // check sub key
                            #ifndef NDEBUG
                            if (d_unique_partition.find(sub_dep) == d_unique_partition.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!unique_partition_finished[sub_dep][sub_n_segment])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            multi_real<<<N_BLOCKS, N_THREADS>>>(
                                _d_unique_partition[0], _d_unique_partition[0],
                                d_unique_partition[sub_dep][sub_n_segment], 1.0, M);
                        }
                        
                        #ifndef NDEBUG
                        unique_partition_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", " << std::to_string(0) << std::endl;
                    }
                }
            }
            // copy jobs that have non-zero segments
            std::vector<std::tuple<std::string, int, int>> parallel_job_copied;
            for (auto it = parallel_job->begin(); it != parallel_job->end(); it++)
            {
                int n_segment_from = std::get<1>(*it);
                int n_segment_to = std::get<2>(*it);
                if(n_segment_to-n_segment_from >= 0)
                    parallel_job_copied.push_back(*it);
            }

            // advance propagator successively
            if(parallel_job_copied.size()==1)
            {
                auto& key = std::get<0>(parallel_job_copied[0]);
                int n_segment_from = std::get<1>(parallel_job_copied[0]);
                int n_segment_to = std::get<2>(parallel_job_copied[0]);
                auto monomer_type = mx->get_unique_branch(key).monomer_type;
                double **_d_unique_partition_key = d_unique_partition[key];

                for(int n=n_segment_from; n<=n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!unique_partition_finished[key][n-1])
                        throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                    #endif

                    one_step_1(
                        _d_unique_partition_key[n-1],
                        _d_unique_partition_key[n],
                        d_boltz_bond[monomer_type],
                        d_boltz_bond_half[monomer_type],
                        d_exp_dw[monomer_type],
                        d_exp_dw_half[monomer_type]);

                    #ifndef NDEBUG
                    unique_partition_finished[key][n] = true;
                    #endif
                }
            }
            else if(parallel_job_copied.size()==2)
            {
                auto& key_1 = std::get<0>(parallel_job_copied[0]);
                int n_segment_from_1 = std::get<1>(parallel_job_copied[0]);
                int n_segment_to_1 = std::get<2>(parallel_job_copied[0]);
                auto species_1 = mx->get_unique_branch(key_1).monomer_type;

                auto& key_2 = std::get<0>(parallel_job_copied[1]);
                int n_segment_from_2 = std::get<1>(parallel_job_copied[1]);
                int n_segment_to_2 = std::get<2>(parallel_job_copied[1]);
                auto species_2 = mx->get_unique_branch(key_2).monomer_type;

                double **_d_unique_partition_key_1 = d_unique_partition[key_1];
                double **_d_unique_partition_key_2 = d_unique_partition[key_2];

                for(int n=0; n<=n_segment_to_1-n_segment_from_1; n++)
                {
                    #ifndef NDEBUG
                    if (!unique_partition_finished[key_1][n-1+n_segment_from_1])
                        throw_with_line_number("unfinished, key: " + key_1 + ", " + std::to_string(n-1+n_segment_from_1));
                    if (!unique_partition_finished[key_2][n-1+n_segment_from_2])
                        throw_with_line_number("unfinished, key: " + key_2 + ", " + std::to_string(n-1+n_segment_from_2));
                    #endif

                    one_step_2(
                        _d_unique_partition_key_1[n-1+n_segment_from_1],
                        _d_unique_partition_key_2[n-1+n_segment_from_2],
                        _d_unique_partition_key_1[n+n_segment_from_1],
                        _d_unique_partition_key_2[n+n_segment_from_2],
                        d_boltz_bond[species_1],
                        d_boltz_bond[species_2],
                        d_boltz_bond_half[species_1],
                        d_boltz_bond_half[species_2],
                        d_exp_dw[species_1],
                        d_exp_dw[species_2],
                        d_exp_dw_half[species_1],
                        d_exp_dw_half[species_2]);

                    #ifndef NDEBUG
                    unique_partition_finished[key_1][n+n_segment_from_1] = true;
                    unique_partition_finished[key_2][n+n_segment_from_2] = true;
                    #endif

                    // std::cout << "finished, key, n: " + key_1 + ", " << std::to_string(n+n_segment_from_1) << std::endl;
                    // std::cout << "finished, key, n: " + key_2 + ", " << std::to_string(n+n_segment_from_2) << std::endl;
                }
            }
            // time_span_count++;
        }

        // compute total partition function of each distinct polymers
        int current_p = 0;
        for(const auto& block: d_unique_phi)
        {
            int p                = std::get<0>(block.first);
            std::string dep_v    = std::get<1>(block.first);
            std::string dep_u    = std::get<2>(block.first);

            // already computed
            if (p != current_p)
                continue;

            int n_superposed;
            // int n_segment_allocated = mx->get_unique_block(block.first).n_segment_allocated;
            int n_segment_offset    = mx->get_unique_block(block.first).n_segment_offset;
            int n_segment_original  = mx->get_unique_block(block.first).n_segment_original;

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_superposed = 1;
            else
                n_superposed = mx->get_unique_block(block.first).v_u.size();

            // check keys
            #ifndef NDEBUG
            if (d_unique_partition.find(dep_v) == d_unique_partition.end())
                throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
            if (d_unique_partition.find(dep_u) == d_unique_partition.end())
                throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");
            #endif

            single_partitions[p] = ((CudaComputationBox *)cb)->inner_product_gpu(
                d_unique_partition[dep_v][n_segment_original-n_segment_offset], // q
                d_unique_partition[dep_u][0])/n_superposed/cb->get_volume();    // q^dagger

            // std::cout << p <<", "<< dep_v <<", "<< dep_u <<", "<< n_segment <<", " << single_partitions[p] << std::endl;
            current_p++;
        }

        // calculate segment concentrations
        for(const auto& block: d_unique_phi)
        {
            int p                = std::get<0>(block.first);
            std::string dep_v    = std::get<1>(block.first);
            std::string dep_u    = std::get<2>(block.first);

            int n_repeated;
            int n_segment_allocated = mx->get_unique_block(block.first).n_segment_allocated;
            int n_segment_offset    = mx->get_unique_block(block.first).n_segment_offset;
            int n_segment_original  = mx->get_unique_block(block.first).n_segment_original;

            // if there is no segment
            if(n_segment_allocated == 0)
            {
                gpu_error_check(cudaMemset(block.second, 0, sizeof(double)*M));
                continue;
            }

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(block.first).v_u.size();
            else
                n_repeated = 1;

            // check keys
            #ifndef NDEBUG
            if (d_unique_partition.find(dep_v) == d_unique_partition.end())
                throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
            if (d_unique_partition.find(dep_u) == d_unique_partition.end())
                throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");
            #endif

            // calculate phi of one block (possibly multiple blocks when using superposition)
            calculate_phi_one_block(
                block.second,               // phi
                d_unique_partition[dep_v],  // dependency v
                d_unique_partition[dep_u],  // dependency u
                n_segment_allocated,
                n_segment_offset,
                n_segment_original);

            // normalize concentration
            PolymerChain& pc = mx->get_polymer(p);
            double norm = mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p]*n_repeated;
            lin_comb<<<N_BLOCKS, N_THREADS>>>(block.second, norm, block.second, 0.0, block.second, M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Advance partial partition function using Richardson extrapolation.
void CudaPseudoContinuous::one_step_1(
    double *d_q_in, double *d_q_out,
    double *d_boltz_bond, double *d_boltz_bond_half,
    double *d_exp_dw, double *d_exp_dw_half)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        //-------------- step 1 ----------
        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step1_1, d_q_in, d_exp_dw, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for_1, d_q_step1_1, d_qk_in_1);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in_1, d_boltz_bond, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak_1, d_qk_in_1, d_q_step1_1);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step1_1, d_q_step1_1, d_exp_dw, 1.0/((double)M), M);

        //-------------- step 2 ----------
        // Evaluate e^(-w*ds/4) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step2_1, d_q_in, d_exp_dw_half, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for_1, d_q_step2_1, d_qk_in_1);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in_1, d_boltz_bond_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak_1, d_qk_in_1, d_q_step2_1);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step2_1, d_q_step2_1, d_exp_dw, 1.0/((double)M), M);
        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for_1, d_q_step2_1, d_qk_in_1);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in_1, d_boltz_bond_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak_1, d_qk_in_1, d_q_step2_1);

        // Evaluate e^(-w*ds/4) in real space.
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step2_1, d_q_step2_1, d_exp_dw_half, 1.0/((double)M), M);
        //-------------- step 3 ----------
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_q_out, 4.0/3.0, d_q_step2_1, -1.0/3.0, d_q_step1_1, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::one_step_2(
    double *d_q_in_1, double *d_q_in_2,
    double *d_q_out_1, double *d_q_out_2,
    double *d_boltz_bond_1, double *d_boltz_bond_2, 
    double *d_boltz_bond_half_1, double *d_boltz_bond_half_2,         
    double *d_exp_dw_1, double *d_exp_dw_2,
    double *d_exp_dw_half_1, double *d_exp_dw_half_2)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        //-------------- step 1 ----------
        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1_2[0], d_q_in_1, d_exp_dw_1, 1.0, M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1_2[M], d_q_in_2, d_exp_dw_2, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for_2, d_q_step1_2, d_qk_in_2);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_2[0],         d_boltz_bond_1, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_2[M_COMPLEX], d_boltz_bond_2, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak_2, d_qk_in_2, d_q_step1_2);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1_2[0], &d_q_step1_2[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1_2[M], &d_q_step1_2[M], d_exp_dw_2, 1.0/((double)M), M);

        //-------------- step 2 ----------
        // Evaluate e^(-w*ds/4) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2_2[0], d_q_in_1, d_exp_dw_half_1, 1.0, M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2_2[M], d_q_in_2, d_exp_dw_half_2, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for_2, d_q_step2_2, d_qk_in_2);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_2[0],         d_boltz_bond_half_1, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_2[M_COMPLEX], d_boltz_bond_half_2, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak_2, d_qk_in_2, d_q_step2_2);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2_2[0], &d_q_step2_2[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2_2[M], &d_q_step2_2[M], d_exp_dw_2, 1.0/((double)M), M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for_2, d_q_step2_2, d_qk_in_2);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_2[0],         d_boltz_bond_half_1, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_2[M_COMPLEX], d_boltz_bond_half_2, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak_2, d_qk_in_2, d_q_step2_2);

        // Evaluate e^(-w*ds/4) in real space.
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2_2[0], &d_q_step2_2[0], d_exp_dw_half_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2_2[M], &d_q_step2_2[M], d_exp_dw_half_2, 1.0/((double)M), M);

        //-------------- step 3 ----------
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_q_out_1, 4.0/3.0, &d_q_step2_2[0], -1.0/3.0, &d_q_step1_2[0], M);
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_q_out_2, 4.0/3.0, &d_q_step2_2[M], -1.0/3.0, &d_q_step1_2[M], M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::calculate_phi_one_block(
    double *d_phi, double **d_q_1, double **d_q_2, const int N, const int N_OFFSET, const int N_ORIGINAL)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        std::vector<double> simpson_rule_coeff = SimpsonQuadrature::get_coeff(N);

        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_ORIGINAL-N_OFFSET], d_q_2[0], simpson_rule_coeff[0], M);
        for(int n=1; n<=N; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_ORIGINAL-N_OFFSET-n], d_q_2[n], simpson_rule_coeff[n], M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaPseudoContinuous::get_total_partition(int polymer)
{
    try
    {
        return single_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::get_monomer_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        // initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // for each block
        for(const auto& block: d_unique_phi)
        {
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = mx->get_unique_block(block.first).n_segment_allocated;
            if (Mixture::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, block.second, M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::get_polymer_concentration(int p, double *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int P = mx->get_n_polymers();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (mx->is_using_superposition())
            throw_with_line_number("Disable 'superposition' option to invoke 'get_polymer_concentration'.");

        PolymerChain& pc = mx->get_polymer(p);
        std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, 1.0, d_unique_phi[std::make_tuple(p, dep_v, dep_u)], M);
            gpu_error_check(cudaMemcpy(&phi[b*M], d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CudaPseudoContinuous::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        auto bond_lengths = mx->get_bond_lengths();
        std::vector<double> stress(cb->get_dim());
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> unique_dq_dl;
        thrust::device_ptr<double> temp_gpu_ptr(d_stress_sum);

        // reset stress map
        for(const auto& item: d_unique_phi)
        {
            for(int d=0; d<3; d++)
                unique_dq_dl[item.first][d] = 0.0;
        }

        // compute stress for unique block
        for(const auto& block: d_unique_phi)
        {
            const auto& key      = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            const int N           = mx->get_unique_block(block.first).n_segment_allocated;
            const int N_OFFSET    = mx->get_unique_block(block.first).n_segment_offset;
            const int N_ORIGINAL  = mx->get_unique_block(block.first).n_segment_original;
            std::string monomer_type = mx->get_unique_block(key).monomer_type;

            // if there is no segment
            if(N == 0)
                continue;

            // contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(block.first).v_u.size();
            else
                n_repeated = 1;

            std::vector<double> s_coeff = SimpsonQuadrature::get_coeff(N);
            double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
            double** d_q_1 = d_unique_partition[dep_v];    // dependency v
            double** d_q_2 = d_unique_partition[dep_u];    // dependency u

            std::array<double,3> _unique_dq_dl = unique_dq_dl[key];

            // compute
            for(int n=0; n<=N; n++)
            {
                // execute a Forward FFT
                gpu_error_check(cudaMemcpy(&d_q_step1_2[0], d_q_1[N_ORIGINAL-N_OFFSET-n], sizeof(double)*M,cudaMemcpyDeviceToDevice));
                gpu_error_check(cudaMemcpy(&d_q_step1_2[M], d_q_2[n], sizeof(double)*M,cudaMemcpyDeviceToDevice));
                cufftExecD2Z(plan_for_2, d_q_step1_2, d_qk_in_2);

                // multiplay two partial partition functions in the fourier spaces
                multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_qk_in_2[0], &d_qk_in_2[M_COMPLEX], M_COMPLEX);
                if ( DIM == 3 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[0] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[1] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[2] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                }
                if ( DIM == 2 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[0] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[1] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
                if ( DIM == 1 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[0] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
            }
            unique_dq_dl[key] = _unique_dq_dl;
        }

        // compute total stress
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] = 0.0;
        for(const auto& block: d_unique_phi)
        {
            const auto& key   = block.first;
            int p             = std::get<0>(key);
            std::string dep_v = std::get<1>(key);
            std::string dep_u = std::get<2>(key);
            PolymerChain& pc  = mx->get_polymer(p);

            for(int d=0; d<cb->get_dim(); d++)
                stress[d] += unique_dq_dl[key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
        }
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/mx->get_ds();
            
        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::get_partial_partition(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        PolymerChain& pc = mx->get_polymer(polymer);
        std::string dep = pc.get_dep(v,u);

        if (mx->get_unique_branches().find(dep) == mx->get_unique_branches().end())
            throw_with_line_number("Could not find the branches '" + dep + "'. Disable 'superposition' option to obtain partial partition functions.");

        const int N = mx->get_unique_branches()[dep].max_n_segment;
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        gpu_error_check(cudaMemcpy(q_out, d_unique_partition[dep][n], sizeof(double)*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}