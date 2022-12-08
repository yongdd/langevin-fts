#include <cmath>
#include "CpuPseudoDiscrete.h"

CpuPseudoDiscrete::CpuPseudoDiscrete(
    ComputationBox *cb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(cb, pc)
{
    try
    {
        const int M = cb->get_n_grid();
        const int N_B = pc->get_n_block();
        const int N = pc->get_n_segment_total();

        this->n_block = N_B;
        this->fft = fft;
        this->boltz_bond = new double*[N_B];
        if (N_B>1)
            this->boltz_bond_middle = new double*[N_B-1];

        this->boltz_bond[0]= new double[n_complex_grid];
        for (int b=0; b<N_B-1; b++)
        {
            this->boltz_bond[b+1]= new double[n_complex_grid];
            this->boltz_bond_middle[b] = new double[n_complex_grid];
        }
        this->q_1 = new double[M*N];
        this->q_2 = new double[M*N];
        
        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuPseudoDiscrete::~CpuPseudoDiscrete()
{
    const int N_B = n_block;

    delete fft;
    delete[] boltz_bond[0];
    for (int b=0; b<N_B-1; b++)
    {
        delete[] boltz_bond[b+1];
        delete[] boltz_bond_middle[b];
    }
    delete[] boltz_bond;
    if (N_B>1)
        delete[] boltz_bond_middle;
    delete[] q_1, q_2;

}
void CpuPseudoDiscrete::update()
{
    try
    {
        double bond_length_middle;
        const int N_B = pc->get_n_block();

        get_boltz_bond(boltz_bond[0],  pc->get_bond_length_sq(0),  cb->get_nx(), cb->get_dx(), pc->get_ds());
        for (int b=0; b<N_B-1; b++)
        {
            bond_length_middle = 0.5*pc->get_bond_length_sq(b) + 0.5*pc->get_bond_length_sq(b+1);
            get_boltz_bond(boltz_bond[b+1],  pc->get_bond_length_sq(b+1),  cb->get_nx(), cb->get_dx(), pc->get_ds());
            get_boltz_bond(boltz_bond_middle[b],  bond_length_middle,  cb->get_nx(), cb->get_dx(), pc->get_ds());
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::array<double,3> CpuPseudoDiscrete::dq_dl()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int N_B  = pc->get_n_block();
        const int N    = pc->get_n_segment_total();
        const int M_COMPLEX = this->n_complex_grid;
        const std::vector<int> seg_start= pc->get_block_start();

        //const double bond_length_middle;// = 0.5*bond_length_a + 0.5*bond_length_b;
        double temp, bond_length_now, *boltz_bond_now;

        std::array<double,3> dq_dl;
        std::complex<double> k_q_1[M_COMPLEX];
        std::complex<double> k_q_2[M_COMPLEX];

        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());

        for(int i=0; i<3; i++)
            dq_dl[i] = 0.0;

        // compute stress within the blocks
        for(int b=0; b<N_B; b++){
            for(int n=seg_start[b]+1; n<seg_start[b+1]; n++){
                bond_length_now = pc->get_bond_length_sq(b);
                boltz_bond_now = boltz_bond[b];

                fft->forward(&q_1[(n-1)*M],k_q_1);
                fft->forward(&q_2[ n   *M],k_q_2);

                if ( DIM >= 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[0] += bond_length_now*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
                }
                if ( DIM >= 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[1] += bond_length_now*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
                }
                if ( DIM >= 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[2] += bond_length_now*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
                }
            }
        }

        // compute stress between the blocks
        for(int b=1; b<N_B; b++){

            int n = seg_start[b];
            bond_length_now = 0.5*pc->get_bond_length_sq(b-1) + 0.5*pc->get_bond_length_sq(b);
            boltz_bond_now = boltz_bond_middle[b-1];

            fft->forward(&q_1[(n-1)*M],k_q_1);
            fft->forward(&q_2[ n   *M],k_q_2);

            if ( DIM >= 3 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[0] += bond_length_now*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
            }
            if ( DIM >= 2 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[1] += bond_length_now*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
            }
            if ( DIM >= 1 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[2] += bond_length_now*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
            }
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
void CpuPseudoDiscrete::compute_statistics(double *phi, double *q_1_init, double *q_2_init,
                                    std::map<std::string, double*> w_block, double &single_partition)
{
    try
    {
        const int  M        = cb->get_n_grid();
        const int  N        = pc->get_n_segment_total();
        const int  N_B      = pc->get_n_block();
        const std::vector<int> N_SEG    = pc->get_n_segment();
        const double ds     = pc->get_ds();
        const std::vector<int> seg_start= pc->get_block_start();

        double exp_dw[N_B][M];

        for(int b=0; b<N_B; b++){
            if( w_block.count(pc->get_type(b)) == 0 )
                throw_with_line_number("block types[" + std::to_string(b) + "] (\"" + pc->get_type(b) + "\") is not in w_block");
            double *w_block_one = w_block[pc->get_type(b)];
            for(int i=0; i<M; i++)
                exp_dw[b][i] = exp(-w_block_one[i]*ds);
        }
        
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                for(int i=0; i<M; i++)
                    q_1[i] = exp_dw[0][i]*q_1_init[i];

                // diffusion of first block segments
                for(int n=1; n<seg_start[1]; n++)
                    one_step(&q_1[(n-1)*M],&q_1[n*M],boltz_bond[0], exp_dw[0]);
                for(int b=1; b<N_B; b++)
                {
                    // diffusion of segment between blocks
                    one_step(&q_1[(seg_start[b]-1)*M],&q_1[seg_start[b]*M],boltz_bond_middle[b-1],exp_dw[b]);
                    // diffusion of b-th block segments
                    for(int n=seg_start[b]+1; n<seg_start[b+1]; n++)
                        one_step(&q_1[(n-1)*M],&q_1[n*M],boltz_bond[b], exp_dw[b]);
                }
            }
            #pragma omp section
            {
                for(int i=0; i<M; i++)
                    q_2[i+(N-1)*M] = exp_dw[N_B-1][i]*q_2_init[i];
                // diffusion of last block segments
                for(int n=N-1; n>seg_start[N_B-1]; n--)
                    one_step(&q_2[n*M],&q_2[(n-1)*M],boltz_bond[N_B-1], exp_dw[N_B-1]);
                for(int b=N_B-1; b>0; b--)
                {
                    // diffusion of segment between blocks
                    one_step(&q_2[seg_start[b]*M],&q_2[(seg_start[b]-1)*M],boltz_bond_middle[b-1],exp_dw[b-1]);
                    // diffusion of (b-1)th block segment
                    for(int n=seg_start[b]-1; n>0; n--)
                        one_step(&q_2[n*M],&q_2[(n-1)*M],boltz_bond[b-1], exp_dw[b-1]);
                }
            }
        }
        // Compute segment concentration of each blocks
        for(int b=0; b<N_B; b++)
        {
            for(int i=0; i<M; i++)
                phi[b*M+i] = q_1[i+seg_start[b]*M]*q_2[i+seg_start[b]*M];
            for(int n=seg_start[b]+1; n<seg_start[b+1]; n++)
            {
                for(int i=0; i<M; i++)
                    phi[b*M+i] += q_1[i+n*M]*q_2[i+n*M];
            }
        }
        // calculates the single chain partition function
        single_partition = cb->inner_product(&q_1[(N-1)*M], q_1_init);

        // normalize the concentration
        for(int b=0; b<N_B; b++)
            for(int i=0; i<M; i++)
                phi[b*M+i] *= cb->get_volume()*pc->get_ds()/exp_dw[b][i]/single_partition;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::one_step(double *q_in, double *q_out,
                                 double *boltz_bond, double *exp_dw)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        std::complex<double> k_q_in[M_COMPLEX];
        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= boltz_bond[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in,q_out);
        // normalization calculation and evaluate e^(-w*ds/2) in real space
        for(int i=0; i<M; i++)
            q_out[i] *= exp_dw[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    // This method should be invoked after invoking compute_statistics().
    
    // Get partial partition functions
    // This is made for debugging and testing.

    const int M = cb->get_n_grid();
    const int N = pc->get_n_segment_total();

    if (n1 < 1 || n1 > N)
        throw_with_line_number("n1 (" + std::to_string(n1) + ") must be in range [1, " + std::to_string(N) + "]");
    if (n2 < 1 || n2 > N)
        throw_with_line_number("n2 (" + std::to_string(n2) + ") must be in range [1, " + std::to_string(N) + "]");

    for(int i=0; i<M; i++)
    {
        q_1_out[i] =q_1[(n1-1)*M+i];
        q_2_out[i] =q_2[(n2-1)*M+i];
    }
}
