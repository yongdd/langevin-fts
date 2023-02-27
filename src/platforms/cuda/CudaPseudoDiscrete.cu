#define THRUST_IGNORE_DEPRECATED_CPP_DIALECToptimal
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "CudaPseudoDiscrete.h"
#include "CudaComputationBox.h"

CudaPseudoDiscrete::CudaPseudoDiscrete(
    ComputationBox *cb,
    Mixture *mx)
    : Pseudo(cb, mx)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // allocate memory for partition functions
        if( mx->get_unique_branches().size() == 0)
            throw_with_line_number("There is no unique branch. Add polymers first.");
        for(const auto& item: mx->get_unique_branches())
        {
            std::string dep = item.first;
            int max_n_segment = item.second.max_n_segment;
             // There are N segments

             // Illustration (N==5)
             // O--O--O--O--O
             // 0  1  2  3  4 unique_blocks

             // Legend)
             // -- : full bond
             // O  : full segment
            d_unique_partition[dep] = new double*[max_n_segment];
            d_unique_partition_size[dep] = max_n_segment;
            for(int i=0; i<d_unique_partition_size[dep]; i++)
                gpu_error_check(cudaMalloc((void**)&d_unique_partition[dep][i], sizeof(double)*M));

            #ifndef NDEBUG
            unique_partition_finished[dep] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                unique_partition_finished[dep][i] = false;
            #endif
        }

        // allocate memory for unique_q_junctions, which contain partition function at junction of discrete chain
        for(const auto& item: mx->get_unique_branches())
        {
            std::string dep = item.first;
            d_unique_q_junctions[dep] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_unique_q_junctions[dep], sizeof(double)*M));
        }

        // allocate memory for concentrations
        if( mx->get_unique_blocks().size() == 0)
            throw_with_line_number("There is no unique block. Add polymers first.");
        for(const auto& item: mx->get_unique_blocks())
        {
            d_unique_phi[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_unique_phi[item.first], sizeof(double)*M));
        }

        // create boltz_bond, boltz_bond_half, and exp_dw
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            d_boltz_bond     [monomer_type] = nullptr;
            d_boltz_bond_half[monomer_type] = nullptr;
            d_exp_dw         [monomer_type] = nullptr;

            gpu_error_check(cudaMalloc((void**)&d_boltz_bond     [monomer_type], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[monomer_type], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw         [monomer_type], sizeof(double)*M));
        }

        // total partition functions for each polymer
        single_partitions = new double[mx->get_n_polymers()];

        // create scheduler for computation of partial partition function
        sc = new Scheduler(mx->get_unique_branches(), N_STREAM); 

        // Create FFT plan
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
        gpu_error_check(cudaMalloc((void**)&d_qk_in_1, sizeof(ftsComplex)*M_COMPLEX));

        gpu_error_check(cudaMalloc((void**)&d_qk_in_2, sizeof(ftsComplex)*2*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q_in_temp_2, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_q_out_temp_2, sizeof(double)*2*M));

        gpu_error_check(cudaMalloc((void**)&d_q_half_step, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_junction,  sizeof(ftsComplex)*M_COMPLEX));
        
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
CudaPseudoDiscrete::~CudaPseudoDiscrete()
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
    for(const auto& item: d_unique_partition)
    {
        for(int i=0; i<d_unique_partition_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: d_unique_phi)
        cudaFree(item.second);
    for(const auto& item: d_unique_q_junctions)
        cudaFree(item.second);

    #ifndef NDEBUG
    for(const auto& item: unique_partition_finished)
        delete[] item.second;
    #endif

    // for get_concentration
    cudaFree(d_phi);

    // for pseudo-spectral: one_step()
    cudaFree(d_qk_in_1);

    cudaFree(d_qk_in_2);
    cudaFree(d_q_in_temp_2);
    cudaFree(d_q_out_temp_2);
    
    cudaFree(d_q_half_step);
    cudaFree(d_q_junction);

    // for stress calculation: compute_stress()
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    cudaFree(d_qk_1);
    cudaFree(d_qk_2);
    cudaFree(d_q_multi);
    cudaFree(d_stress_sum);
}

void CudaPseudoDiscrete::update_bond_function()
{
    try
    {
        // for pseudo-spectral: one_step()
        const int M_COMPLEX = this->n_complex_grid;
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];
        
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;

            get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), mx->get_ds());
            get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), mx->get_ds());

            gpu_error_check(cudaMemcpy(d_boltz_bond     [monomer_type], boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
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
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::compute_statistics(
    std::map<std::string, double*> w_input,
    std::map<std::string, double*> q_init)
{
    try
    {
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

        // exp_dw
        double exp_dw[M];
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
                exp_dw[i] = exp(-w[i]*ds);
            gpu_error_check(cudaMemcpy(d_exp_dw[monomer_type], exp_dw, sizeof(double)*M,cudaMemcpyHostToDevice));
        }

        double q_uniform[M];
        for(int i=0; i<M; i++)
            q_uniform[i] = 1.0;

        // for each time span
        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // multiplay all partition functions at junctions if necessary 
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

                // calculate one block end
                if(n_segment_from == 1 && deps.size() == 0) // if it is leaf node
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = Mixture::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(_d_unique_partition[0], q_init[g], sizeof(double)*M, cudaMemcpyHostToDevice));
                        multi_real<<<N_BLOCKS, N_THREADS>>>(_d_unique_partition[0], _d_unique_partition[0], d_exp_dw[monomer_type], 1.0, M);
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(_d_unique_partition[0], d_exp_dw[monomer_type], sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    }
                    
                    #ifndef NDEBUG
                    unique_partition_finished[key][0] = true;
                    #endif
                }
                else if (n_segment_from == 1 && deps.size() > 0) // if it is not leaf node
                {
                    // if it is superposed
                    if (key[0] == '[')
                    {
                        // initialize to zero
                        gpu_error_check(cudaMemset(_d_unique_partition[0], 0, sizeof(double)*M));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // check sub key
                            #ifndef NDEBUG
                            if (d_unique_partition.find(sub_dep) == d_unique_partition.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!unique_partition_finished[sub_dep][sub_n_segment-1])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            lin_comb<<<N_BLOCKS, N_THREADS>>>(
                                _d_unique_partition[0], 1.0, _d_unique_partition[0],
                                sub_n_repeated, d_unique_partition[sub_dep][sub_n_segment-1], M);
                        }
                        one_step_1(_d_unique_partition[0],
                               _d_unique_partition[0],
                               d_boltz_bond[monomer_type],
                               d_exp_dw[monomer_type]);   

                        #ifndef NDEBUG
                        unique_partition_finished[key][0] = true;
                        #endif
                    }
                    else
                    {
                                                // Illustration (four branches)
                        //     A
                        //     |
                        // O - . - B
                        //     |
                        //     C

                        // Legend)
                        // .       : junction
                        // O       : full segment
                        // -, |    : half bonds
                        // A, B, C : other full segments

                        // combine branches
                        gpu_error_check(cudaMemcpy(d_q_junction, q_uniform, sizeof(double)*M, cudaMemcpyHostToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // check sub key
                            #ifndef NDEBUG
                            if (d_unique_partition.find(sub_dep) == d_unique_partition.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!unique_partition_finished[sub_dep][sub_n_segment-1])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            half_bond_step(d_unique_partition[sub_dep][sub_n_segment-1],
                                d_q_half_step, d_boltz_bond_half[mx->get_unique_branch(sub_dep).monomer_type]);

                            multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_junction, d_q_junction, d_q_half_step, 1.0, M);
                        }
                        gpu_error_check(cudaMemcpy(d_unique_q_junctions[key], d_q_junction, sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        // add half bond
                        half_bond_step(d_q_junction, _d_unique_partition[0], d_boltz_bond_half[monomer_type]);

                        // add full segment
                        multi_real<<<N_BLOCKS, N_THREADS>>>(_d_unique_partition[0], _d_unique_partition[0], d_exp_dw[monomer_type], 1.0, M);

                        #ifndef NDEBUG
                        unique_partition_finished[key][0] = true;
                        #endif
                    }
                }
                else
                {
                    int n = n_segment_from-1;

                    #ifndef NDEBUG
                    if (!unique_partition_finished[key][n-1])
                        throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                    #endif

                    one_step_1(_d_unique_partition[n-1],
                               _d_unique_partition[n],
                               d_boltz_bond[monomer_type],
                               d_exp_dw[monomer_type]);

                    #ifndef NDEBUG
                    unique_partition_finished[key][n] = true;
                    #endif
                }
            }
                
            // apply the propagator successively
            if(parallel_job->size()==1)
            {
                auto& key = std::get<0>((*parallel_job)[0]);
                int n_segment_from = std::get<1>((*parallel_job)[0]);
                int n_segment_to = std::get<2>((*parallel_job)[0]);
                auto monomer_type = mx->get_unique_branch(key).monomer_type;
                double **_d_unique_partition_key = d_unique_partition[key];

                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!unique_partition_finished[key][n-1])
                        throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                    #endif

                    one_step_1(_d_unique_partition_key[n-1],
                            _d_unique_partition_key[n],
                            d_boltz_bond[monomer_type],
                            d_exp_dw[monomer_type]);

                    #ifndef NDEBUG
                    unique_partition_finished[key][n] = true;
                    #endif
                }
            }
            else if(parallel_job->size()==2)
            {
                auto& key_1 = std::get<0>((*parallel_job)[0]);
                int n_segment_from_1 = std::get<1>((*parallel_job)[0]);
                int n_segment_to_1 = std::get<2>((*parallel_job)[0]);
                auto species_1 = mx->get_unique_branch(key_1).monomer_type;

                auto& key_2 = std::get<0>((*parallel_job)[1]);
                int n_segment_from_2 = std::get<1>((*parallel_job)[1]);
                int n_segment_to_2 = std::get<2>((*parallel_job)[1]);
                auto species_2 = mx->get_unique_branch(key_2).monomer_type;

                double **_d_unique_partition_key_1 = d_unique_partition[key_1];
                double **_d_unique_partition_key_2 = d_unique_partition[key_2];

                for(int n=0; n<n_segment_to_1-n_segment_from_1; n++)
                {
                    #ifndef NDEBUG
                    if (!unique_partition_finished[key_1][n-1+n_segment_from_1])
                        throw_with_line_number("unfinished, key: " + key_1 + ", " + std::to_string(n-n_segment_from_1));
                    if (!unique_partition_finished[key_2][n-1+n_segment_from_2])
                        throw_with_line_number("unfinished, key: " + key_2 + ", " + std::to_string(n-n_segment_from_2));
                    #endif

                    one_step_2(
                        _d_unique_partition_key_1[n-1+n_segment_from_1],
                        _d_unique_partition_key_2[n-1+n_segment_from_2],
                        _d_unique_partition_key_1[n+n_segment_from_1],
                        _d_unique_partition_key_2[n+n_segment_from_2],
                        d_boltz_bond[species_1],
                        d_boltz_bond[species_2],
                        d_exp_dw[species_1],
                        d_exp_dw[species_2]);

                    #ifndef NDEBUG
                    unique_partition_finished[key_1][n+n_segment_from_1] = true;
                    unique_partition_finished[key_2][n+n_segment_from_2] = true;
                    #endif

                    // std::cout << "finished, key, n: " + key_1 + ", " << std::to_string(n+n_segment_from_1) << std::endl;
                    // std::cout << "finished, key, n: " + key_2 + ", " << std::to_string(n+n_segment_from_2) << std::endl;
                }
            }
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
            std::string monomer_type = mx->get_unique_block(block.first).monomer_type;

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

            single_partitions[p] = ((CudaComputationBox *)cb)->inner_product_inverse_weight_gpu(
                d_unique_partition[dep_v][n_segment_original-n_segment_offset-1],  // q
                d_unique_partition[dep_u][0],                                      // q^dagger
                d_exp_dw[monomer_type])/n_superposed/cb->get_volume();        

            // std::cout << p <<", "<< dep_v <<", "<< dep_u <<", "<< n_segment <<", " << single_partitions[p] << std::endl;
            // std::cout << p <<", "<< n_segment <<", "<< n_segment_offset <<", "<< single_partitions[p] << std::endl;
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
            std::string monomer_type = mx->get_unique_block(block.first).monomer_type;

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
                block.second,             // phi
                d_unique_partition[dep_v],  // dependency v
                d_unique_partition[dep_u],  // dependency u
                d_exp_dw[monomer_type],     // exp_dw
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
void CudaPseudoDiscrete::one_step_1(
    double *d_q_in, double *d_q_out,
    double *d_boltz_bond, double *d_exp_dw)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        //-------------- step 1 ----------
        // Execute a Forward FFT
        cufftExecD2Z(plan_for_1, d_q_in, d_qk_in_1);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in_1, d_boltz_bond, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_1, d_qk_in_1, d_q_out);

        // Evaluate e^(-w*ds) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_out, d_q_out, d_exp_dw, 1.0/((double)M), M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::one_step_2(
    double *d_q_in_1, double *d_q_in_2,
    double *d_q_out_1, double *d_q_out_2,
    double *d_boltz_bond_1, double *d_boltz_bond_2,  
    double *d_exp_dw_1, double *d_exp_dw_2)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        gpu_error_check(cudaMemcpy(&d_q_in_temp_2[0], d_q_in_1, sizeof(double)*M, cudaMemcpyDeviceToDevice));
        gpu_error_check(cudaMemcpy(&d_q_in_temp_2[M], d_q_in_2, sizeof(double)*M, cudaMemcpyDeviceToDevice));

        //-------------- step 1 ----------
        // Execute a Forward FFT
        cufftExecD2Z(plan_for_2, d_q_in_temp_2, d_qk_in_2);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_2[0],         d_boltz_bond_1, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_2[M_COMPLEX], d_boltz_bond_2, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_2, d_qk_in_2, d_q_out_temp_2);

        // Evaluate e^(-w*ds) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_out_1, &d_q_out_temp_2[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_out_2, &d_q_out_temp_2[M], d_exp_dw_2, 1.0/((double)M), M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::half_bond_step(double *d_q_in, double *d_q_out, double *d_boltz_bond_half)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // 3D fourier discrete transform, forward and inplace
        cufftExecD2Z(plan_for_1, d_q_in, d_qk_in_1);
        // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in_1, d_boltz_bond_half, 1.0/((double)M), M_COMPLEX);
        // 3D fourier discrete transform, backward and inplace
        cufftExecZ2D(plan_bak_1, d_qk_in_1, d_q_out);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::calculate_phi_one_block(
    double *d_phi, double **d_q_1, double **d_q_2, double *d_exp_dw, const int N, const int N_OFFSET, const int N_ORIGINAL)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi,d_q_1[N_ORIGINAL-N_OFFSET-1], d_q_2[0], 1.0, M);
        for(int n=1; n<N; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_ORIGINAL-N_OFFSET-n-1], d_q_2[n], 1.0, M);
        }
        divide_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi, d_exp_dw, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaPseudoDiscrete::get_total_partition(int polymer)
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
void CudaPseudoDiscrete::get_monomer_concentration(std::string monomer_type, double *phi)
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
void CudaPseudoDiscrete::get_polymer_concentration(int p, double *phi)
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
            throw_with_line_number("Disable 'superposition' option to obtain concentration of each block.");

        PolymerChain& pc = mx->get_polymer(p);
        std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            // copy normalized concentration
            double norm = cb->get_volume()*mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, norm, d_unique_phi[std::make_tuple(p, dep_v, dep_u)], M);
            gpu_error_check(cudaMemcpy(&phi[b*M], d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CudaPseudoDiscrete::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.
    try
    {
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

            // contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(block.first).v_u.size();
            else
                n_repeated = 1;

            double **d_q_1 = d_unique_partition[dep_v];    // dependency v
            double **d_q_2 = d_unique_partition[dep_u];    // dependency u

            double bond_length_sq;
            double *d_boltz_bond_now;

            std::array<double,3> _unique_dq_dl = unique_dq_dl[key];

            // std::cout << "dep_v: " << dep_v << std::endl;
            // std::cout << "dep_u: " << dep_u << std::endl;

            // compute stress
            for(int n=0; n<=N; n++)
            {
                // at v
                if (n + N_OFFSET == N_ORIGINAL)
                {
                    // std::cout << "case 1: " << unique_q_junctions[dep_v][0] << ", " << q_2[(N-1)*M] << std::endl;
                    if (mx->get_unique_branch(dep_v).deps.size() == 0) // if v is leaf node, skip
                        continue;
                    
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[0], d_unique_q_junctions[dep_v], sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[M], d_q_2[N-1],                  sizeof(double)*M, cudaMemcpyDeviceToDevice));

                    bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    d_boltz_bond_now = d_boltz_bond_half[monomer_type];
                }
                // at u
                else if (n + N_OFFSET == 0){
                    // std::cout << "case 2: " << q_1[(N_ORIGINAL-N_OFFSET-1)*M] << ", " << unique_q_junctions[dep_u][0] << std::endl;
                    if (mx->get_unique_branch(dep_u).deps.size() == 0) // if u is leaf node, skip
                        continue;

                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[0], d_q_1[N_ORIGINAL-1],         sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[M], d_unique_q_junctions[dep_u], sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    d_boltz_bond_now = d_boltz_bond_half[monomer_type];
                }
                // at superposition junction
                else if (n == 0)
                {
                    // std::cout << "case 4" << std::endl;
                    continue;
                }
                // within the blocks
                else
                {
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[0], d_q_1[N_ORIGINAL-N_OFFSET-n-1], sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[M], d_q_2[n-1],                     sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    d_boltz_bond_now = d_boltz_bond[monomer_type];
                }

                // execute a Forward FFT
                cufftExecD2Z(plan_for_2, d_q_in_temp_2, d_qk_in_2);

                // multiplay two partial partition functions in the fourier spaces
                multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_qk_in_2[0], &d_qk_in_2[M_COMPLEX], M_COMPLEX);

                multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_multi, d_q_multi, d_boltz_bond_now, bond_length_sq, M_COMPLEX);
                if ( DIM == 3 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, 1.0, M_COMPLEX);
                    _unique_dq_dl[0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, 1.0, M_COMPLEX);
                    _unique_dq_dl[1] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0, M_COMPLEX);
                    _unique_dq_dl[2] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
                if ( DIM == 2 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, 1.0, M_COMPLEX);
                    _unique_dq_dl[0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0, M_COMPLEX);
                    _unique_dq_dl[1] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
                if ( DIM == 1 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0, M_COMPLEX);
                    _unique_dq_dl[0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
            }
            unique_dq_dl[key] = _unique_dq_dl;
        }

        // compute total stress
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] = 0.0;
        for(const auto& block: d_unique_phi)
        {
            const auto& key      = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);
            PolymerChain& pc = mx->get_polymer(p);

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
void CudaPseudoDiscrete::get_partial_partition(double *q_out, int polymer, int v, int u, int n)
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
        if (n < 1 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N) + "]");

        gpu_error_check(cudaMemcpy(q_out, d_unique_partition[dep][n-1], sizeof(double)*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
