#define THRUST_IGNORE_DEPRECATED_CPP_DIALECToptimal
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "CudaPseudoBranchedDiscrete.h"
#include "CudaComputationBox.h"

CudaPseudoBranchedDiscrete::CudaPseudoBranchedDiscrete(
    ComputationBox *cb,
    PolymerChain *pc)
    : Pseudo(cb, pc)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // create reduced_edges, which contain partition function
        for(const auto& item: pc->get_reduced_branches_max_segment())
        {
            std::string dep = item.first;
            d_reduced_edges[dep].max_n_segment = item.second;
             // There are N segments

             // Illustration (N==5)
             // O--O--O--O--O
             // 0  1  2  3  4 

             // Legend)
             // -- : full bond
             // O  : full segment
             
            d_reduced_edges[dep].species = pc->key_to_species(dep);
            d_reduced_edges[dep].deps    = pc->key_to_deps(dep);
            d_reduced_edges[dep].partition = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_reduced_edges[dep].partition, sizeof(double)*M*item.second));
        }

        // create reduced_q_junctions, which contain partition function at junction of discrete chain
        for(const auto& item: pc->get_reduced_branches_max_segment())
        {
            std::string dep = item.first;
            d_reduced_q_junctions[dep] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_reduced_q_junctions[dep], sizeof(double)*M));
        }

        // create reduced_blocks, which contains concentration
        std::vector<polymer_chain_block>& blocks = pc->get_blocks();
        for(int i=0; i<blocks.size(); i++)
        {
            std::string dep_v = pc->get_dep(blocks[i].v, blocks[i].u);
            std::string dep_u = pc->get_dep(blocks[i].u, blocks[i].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);
            std::pair<std::string, std::string> key = std::make_pair(dep_v, dep_u);
            d_reduced_blocks[key].n_segment = blocks[i].n_segment;
            d_reduced_blocks[key].species   = blocks[i].species;
        }
        for(const auto& item: d_reduced_blocks)
        {
            //std::cout << "reduced_blocks: " << dep_v << ", " << dep_u << std::endl;
            d_reduced_blocks[item.first].phi = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_reduced_blocks[item.first].phi, sizeof(double)*M));
        }

        // create boltz_bond, boltz_bond_half, and exp_dw
        for(const auto& item: pc->get_dict_bond_lengths())
        {
            std::string species = item.first;
            d_boltz_bond     [species] = nullptr;
            d_boltz_bond_half[species] = nullptr;
            d_exp_dw         [species] = nullptr;

            gpu_error_check(cudaMalloc((void**)&d_boltz_bond     [species], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[species], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw         [species], sizeof(double)*M));
        }

        // Create FFT plan
        const int BATCH{1};
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
            n_grid[0] = cb->get_nx(1);
            n_grid[1] = cb->get_nx(2);
        }
        else if(cb->get_dim() == 1)
        {
            n_grid[0] = cb->get_nx(2);
        }
        cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,BATCH);
        cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,BATCH);

        // allocate memory for pseudo-spectral: one-step()
        gpu_error_check(cudaMalloc((void**)&d_qk_in, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_half_step, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_junction,  sizeof(ftsComplex)*M_COMPLEX));
        
        // allocate memory for stress calculation: dq_dl()
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_qk_1,        sizeof(ftsComplex)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_qk_2,        sizeof(ftsComplex)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q_multi,         sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_stress_sum,      sizeof(double)*M_COMPLEX));

        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoBranchedDiscrete::~CudaPseudoBranchedDiscrete()
{
    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    for(const auto& item: d_boltz_bond)
        cudaFree(item.second);
    for(const auto& item: d_boltz_bond_half)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw)
        cudaFree(item.second);
    for(const auto& item: d_reduced_edges)
        cudaFree(item.second.partition);
    for(const auto& item: d_reduced_blocks)
        cudaFree(item.second.phi);
    for(const auto& item: d_reduced_q_junctions)
        cudaFree(item.second);

    // for pseudo-spectral: one-step()
    cudaFree(d_qk_in);
    cudaFree(d_q_half_step);
    cudaFree(d_q_junction);

    // for stress calculation: dq_dl()
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    cudaFree(d_qk_1);
    cudaFree(d_qk_2);
    cudaFree(d_q_multi);
    cudaFree(d_stress_sum);
}

void CudaPseudoBranchedDiscrete::update()
{
    try
    {
        // for pseudo-spectral: one-step()
        const int N_B = pc->get_n_block();
        const int M_COMPLEX = this->n_complex_grid;
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];
        
        for(const auto& item: pc->get_dict_bond_lengths())
        {
            std::string species = item.first;
            double bond_length_sq = item.second*item.second;

            get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), pc->get_ds());
            get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), pc->get_ds());

            gpu_error_check(cudaMemcpy(d_boltz_bond     [species], boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_boltz_bond_half[species], boltz_bond_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        }

        // for stress calculation: dq_dl()
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
void CudaPseudoBranchedDiscrete::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_block,
    double *phi, double &single_partition)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const double ds = pc->get_ds();

        for(const auto& item: d_reduced_edges)
        {
            if( w_block.count(item.second.species) == 0)
                throw_with_line_number("\"" + item.second.species + "\" species is not in w_block.");
        }

        if( q_init.size() > 0)
            throw_with_line_number("Currently, \'q_init\' is not supported for branched polymers.");

        // exp_dw
        double exp_dw[M];
        for(const auto& item: w_block)
        {
            std::string species = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
                exp_dw[i] = exp(-w[i]*ds);
            gpu_error_check(cudaMemcpy(d_exp_dw[species], exp_dw, sizeof(double)*M,cudaMemcpyHostToDevice));
        }

        double q_uniform[M];
        for(int i=0; i<M; i++)
            q_uniform[i] = 1.0;
        for(const auto& item: d_reduced_edges)
        {
            // calculate one block end
            if (item.second.deps.size() > 0) // if it is not leaf node
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
                gpu_error_check(cudaMemcpy(d_q_junction, q_uniform, sizeof(double)*M,cudaMemcpyHostToDevice));

                for(int p=0; p<item.second.deps.size(); p++)
                {
                    std::string sub_dep = item.second.deps[p].first;
                    int sub_n_segment   = item.second.deps[p].second;

                    half_bond_step(&d_reduced_edges[sub_dep].partition[(sub_n_segment-1)*M],
                        d_q_half_step, d_boltz_bond_half[d_reduced_edges[sub_dep].species]);

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_junction, d_q_junction, d_q_half_step, 1.0, M);
                }
                gpu_error_check(cudaMemcpy(d_reduced_q_junctions[item.first], d_q_junction, sizeof(double)*M,cudaMemcpyDeviceToDevice));

                // add half bond
                half_bond_step(d_reduced_q_junctions[item.first], &item.second.partition[0], d_boltz_bond_half[item.second.species]);

                // add full segment
                multi_real<<<N_BLOCKS, N_THREADS>>>(&item.second.partition[0], &item.second.partition[0], d_exp_dw[item.second.species], 1.0, M);
            }
            else  // if it is leaf node
            {
                //* q_init
                gpu_error_check(cudaMemcpy(&item.second.partition[0], d_exp_dw[item.second.species], sizeof(double)*M,cudaMemcpyDeviceToDevice));
            }

            // diffusion of each blocks
            for(int n=1; n<item.second.max_n_segment; n++)
            {
                one_step(&item.second.partition[(n-1)*M],
                         &item.second.partition[n*M],
                         d_boltz_bond[item.second.species],
                         d_exp_dw[item.second.species]);
            }
        }

        // calculate the single chain partition function
        std::string dep_v = d_reduced_blocks.begin()->first.first;
        std::string dep_u = d_reduced_blocks.begin()->first.second;
        int n_segment = d_reduced_blocks.begin()->second.n_segment;
        single_partition = ((CudaComputationBox *)cb)->inner_product_inverse_weight_gpu(
            &d_reduced_edges[dep_v].partition[(n_segment-1)*M],  // q
            &d_reduced_edges[dep_u].partition[0],                // q^dagger
            d_exp_dw[d_reduced_blocks.begin()->second.species]);

        // calculate segment concentration
        for(const auto& item: d_reduced_blocks)
        {
            calculate_phi_one_type(
                item.second.phi,                                // phi
                d_reduced_edges[item.first.first].partition,    // dependency v
                d_reduced_edges[item.first.second].partition,   // dependency u
                item.second.n_segment);                         // n_segment
        }

        // normalize the concentration
        for(const auto& item: d_reduced_blocks)
            divide_real<<<N_BLOCKS, N_THREADS>>>(item.second.phi, item.second.phi,
            d_exp_dw[item.second.species], cb->get_volume()*pc->get_ds()/single_partition, M);

        // copy phi
        std::vector<polymer_chain_block>& blocks = pc->get_blocks();
        for(int n=0; n<blocks.size(); n++)
        {
            std::string dep_v = pc->get_dep(blocks[n].v, blocks[n].u);
            std::string dep_u = pc->get_dep(blocks[n].u, blocks[n].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);
            gpu_error_check(cudaMemcpy(
                &phi[n*M],d_reduced_blocks[std::make_pair(dep_v, dep_u)].phi,
                sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaPseudoBranchedDiscrete::one_step(
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
        cufftExecD2Z(plan_for, d_q_in, d_qk_in);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in, d_boltz_bond, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak, d_qk_in, d_q_out);

        // Evaluate e^(-w*ds) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_out, d_q_out, d_exp_dw, 1.0/((double)M), M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoBranchedDiscrete::half_bond_step(double *d_q_in, double *d_q_out, double *d_boltz_bond_half)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // 3D fourier discrete transform, forward and inplace
        cufftExecD2Z(plan_for, d_q_in, d_qk_in);
        // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in, d_boltz_bond_half, 1.0/((double)M), M_COMPLEX);
        // 3D fourier discrete transform, backward and inplace
        cufftExecZ2D(plan_bak, d_qk_in, d_q_out);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoBranchedDiscrete::calculate_phi_one_type(
    double *d_phi, double *d_q_1, double *d_q_2, const int N)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, &d_q_1[0], &d_q_2[(N-1)*M], 1.0, M);
        for(int n=1; n<N; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, &d_q_1[n*M], &d_q_2[(N-n-1)*M], 1.0, M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::array<double,3> CudaPseudoBranchedDiscrete::dq_dl()
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

        std::array<double,3> dq_dl;
        std::map<std::pair<std::string, std::string>, std::array<double,3>> reduced_dq_dl;
        std::map<std::string, double>& dict_bond_lengths = pc->get_dict_bond_lengths();
        thrust::device_ptr<double> temp_gpu_ptr(d_stress_sum);

        // compute stress for reduced key pairs
        for(const auto& item: d_reduced_blocks)
        {
            const int N = item.second.n_segment;
            auto key = item.first;
            std::string dep_v = key.first;
            std::string dep_u = key.second; 
            std::string species = item.second.species;
            double* d_q_1 = d_reduced_edges[dep_v].partition;   // dependency v
            double* d_q_2 = d_reduced_edges[dep_u].partition;   // dependency u

            double bond_length_sq;
            double *d_boltz_bond_now;

            // reset
            for(int d=0; d<3; d++)
                reduced_dq_dl[key][d] = 0.0;

            // std::cout << "dep_v: " << dep_v << std::endl;
            // std::cout << "dep_u: " << dep_u << std::endl;

            // compute stress
            for(int n=0; n<=N; n++)
            {
                std::cout<< "reduced_dq_dl[key][0]: " << reduced_dq_dl[key][0] << std::endl;
                // at v
                if (n == 0){
                    if (d_reduced_edges[dep_v].deps.size() == 0) // if v is leaf node, skip
                        continue;
                    cufftExecD2Z(plan_for, d_reduced_q_junctions[dep_v], d_qk_1);
                    cufftExecD2Z(plan_for, &d_q_2[(N-1)*M],              d_qk_2);
                    bond_length_sq = 0.5*dict_bond_lengths[species]*dict_bond_lengths[species];
                    d_boltz_bond_now = d_boltz_bond_half[species];
                }
                // at u  
                else if (n == N)
                {
                    if (d_reduced_edges[dep_u].deps.size() == 0) // if u is leaf node, skip
                        continue; 
                    cufftExecD2Z(plan_for, &d_q_1[(N-1)*M],              d_qk_1);
                    cufftExecD2Z(plan_for, d_reduced_q_junctions[dep_u], d_qk_2);
                    bond_length_sq = 0.5*dict_bond_lengths[species]*dict_bond_lengths[species];
                    d_boltz_bond_now = d_boltz_bond_half[species];
                }
                // within the blocks
                else
                {
                    cufftExecD2Z(plan_for, &d_q_1[(n-1)*M],   d_qk_1);
                    cufftExecD2Z(plan_for, &d_q_2[(N-n-1)*M], d_qk_2);
                    bond_length_sq = dict_bond_lengths[species]*dict_bond_lengths[species];
                    d_boltz_bond_now = d_boltz_bond[species];
                }

                // compute
                multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, d_qk_1, d_qk_2, M_COMPLEX);
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_multi, d_q_multi, d_boltz_bond_now, bond_length_sq, M_COMPLEX);
                if ( DIM >= 3 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, 1.0, M_COMPLEX);
                    reduced_dq_dl[key][0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
                if ( DIM >= 2 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, 1.0, M_COMPLEX);
                    reduced_dq_dl[key][1] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
                if ( DIM >= 1 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0, M_COMPLEX);
                    reduced_dq_dl[key][2] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
            }
        }

        // compute total stress
        std::vector<polymer_chain_block>& blocks = pc->get_blocks();
        for(int d=0; d<3; d++)
            dq_dl[d] = 0.0;
        for(int n=0; n<blocks.size(); n++)
        {
            std::string dep_v = pc->get_dep(blocks[n].v, blocks[n].u);
            std::string dep_u = pc->get_dep(blocks[n].u, blocks[n].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);
            for(int d=0; d<3; d++)
                dq_dl[d] += reduced_dq_dl[std::make_pair(dep_v, dep_u)][d];
        }
        for(int d=0; d<3; d++)
            dq_dl[d] /= 3.0*cb->get_lx(d)*M*M/pc->get_ds()/cb->get_volume();

        return dq_dl;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoBranchedDiscrete::get_partition(double *q_out, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        std::string dep = pc->get_dep(v,u);
        const int N = d_reduced_edges[dep].max_n_segment;
        if (n < 1 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N) + "]");

        double* d_partition = d_reduced_edges[dep].partition;
        gpu_error_check(cudaMemcpy(q_out, &d_partition[(n-1)*M], sizeof(double)*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
