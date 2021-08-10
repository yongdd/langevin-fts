/*-----------------------------------------------------------------
! This is an Anderson mixing module implmented in C++.
! This version is a little bit faster than Fortran version.
!-----------------------------------------------------------------*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

using namespace std;

typedef double ftsReal;

template <unsigned int blockSize>
__device__ static void warpReduce(volatile ftsReal* sdata, int tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ static void multi_dot_kernel(int ncomp, ftsReal *seg_d, ftsReal *a_d, ftsReal *b_d, ftsReal *sum_d, unsigned int M)
{
    extern __shared__ ftsReal sdata[];
// each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x*gridDim.x;
    sdata[tid] = 0.0;
    while (i < M)
    {
        for(int n = 0; n < ncomp; n++)
        {
            sdata[tid] += seg_d[i]*a_d[i+n*M]*b_d[i+n*M];
        }
        i += gridSize;
    }
    __syncthreads();

// do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid < 64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

// write result for this block to global mem
    if (tid == 0) sum_d[blockIdx.x] = sdata[0];
}

__global__ static void addLinComb(ftsReal* dst,
                        ftsReal a,
                        ftsReal* src1,
                        ftsReal b,
                        ftsReal* src2,
                        const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = dst[i] + a*src1[i] + b*src2[i];
        i += blockDim.x * gridDim.x;
    }
}

/*-----------------------------------------------------------------
! A circular buffer is a data structure that uses a single,
! fixed-size buffer as if it were connected end-to-end.
! Each elements are 1-dmensional real array.
!-----------------------------------------------------------------*/
class CircularBuffer
{
private:
    int length; // maximum number of elements
    int width;  // size of each elements
    int start;  // index of oldest elements
    int n_items;   // index at which to write new element
    ftsReal** elems;

public:
    CircularBuffer(int length, int width)
    {
        this->length = length;
        this->width = width;
        this->start = 0;
        this->n_items = 0;

        elems = new ftsReal*[length];
        for (int i=0; i<length; i++)
            elems[i] = new ftsReal[width];
    }
    ~CircularBuffer()
    {
        for (int i=0; i<length; i++)
            delete[] elems[i];
        delete[] elems;
    }
    void reset()
    {
        start = 0;
        n_items = 0;
    }
    void insert(ftsReal* new_arr)
    {
        for(int i=0; i<width; i++)
            elems[(start+n_items)%length][i] = new_arr[i];
        if (n_items == length)
            start = (start+1)%length;
        n_items = min(n_items+1, length);

    }
    ftsReal* getArray(int n)
    {
        return elems[(start+n)%length];
    }
    ftsReal get(int n, int m)
    {
        return elems[(start+n)%length][m];
    }
    ftsReal getSym(int n, int m)
    {
        return elems[(start+max(n,m))%length][abs(n-m)];
    }
};
/*-----------------------------------------------------------------
! A circular buffer stores data in the GPU memory.
!-----------------------------------------------------------------*/
class CircularBufferCUDA
{
private:
    int length; // maximum number of elements
    int width;  // size of each elements
    int start;  // index of oldest elements
    int n_items;   // index at which to write new element
    ftsReal** elems_d;

public:
    CircularBufferCUDA(int length, int width)
    {
        this->length = length;
        this->width = width;
        this->start = 0;
        this->n_items = 0;

        elems_d = new ftsReal*[length];
        for (int i=0; i<length; i++)
            cudaMalloc((void**)&elems_d[i], sizeof(ftsReal)*width);
    }
    ~CircularBufferCUDA()
    {
        for (int i=0; i<length; i++)
            cudaFree(elems_d[i]);
        delete[] elems_d;
    }
    void reset()
    {
        start = 0;
        n_items = 0;
    }
    void insert(ftsReal* new_arr)
    {
        cudaMemcpy(elems_d[(start+n_items)%length], new_arr,
                   sizeof(ftsReal)*width, cudaMemcpyHostToDevice);
        if (n_items == length)
            start = (start+1)%length;
        n_items = min(n_items+1, length);
    }
    ftsReal* getArray(int n)
    {
        return elems_d[(start+n)%length];
    }
};

/*-------------------------------------------------------------
!  Anderson mixing module
-------------------------------------------------------------*/
/* a few previous field values are stored for anderson mixing */
static CircularBufferCUDA *cb_wout_hist, *cb_wdiff_hist;
/* arrays to calculate anderson mixing */
static CircularBuffer *cb_wdiffdots;
static ftsReal **u_nm, *v_n, *a_n, *wdiffdots;

static int N_BLOCKS;
static int N_THREADS;

static int num_components, totalMM, total_grids;
static ftsReal start_anderson_error, mix_min, mix, init_mix;
static int max_anderson, n_anderson;
static ftsReal *seg, *seg_d;
static ftsReal *temp, *temp_d;
static ftsReal *sum, *sum_d;
static ftsReal *w_diff_d, *w_d;

extern "C" void am_cuda_reset_count_();
static ftsReal multi_dot_cuda(int n_comp, ftsReal *a, ftsReal *b);
static void find_an(ftsReal **u_nm, ftsReal *v_n, ftsReal *a_n, int n);

extern "C" void am_cuda_initialize_(
    int    *p_num_components,
    int    *p_totalMM,
    ftsReal *p_seg,
    int    *p_max_anderson,
    ftsReal *p_start_anderson_error,
    ftsReal *p_mix_min,
    ftsReal *p_init_mix,
    int *pN_BLOCKS,
    int *pN_THREADS,
    int *pProcessIdx)
{
    int device;
    int devicesCount;
    struct cudaDeviceProp prop;

    N_BLOCKS = *pN_BLOCKS;
    N_THREADS = *pN_THREADS;

    cudaGetDeviceCount(&devicesCount);
    cudaSetDevice((*pProcessIdx)%devicesCount);

    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit (1);
    }
    cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit (1);
    }

    if (prop.warpSize < 32)
    {
        printf("'Warp size' cannot be less than 32 due to synchronization in 'multi_dot_kernel'.\n");
        exit (1);
    }

    if (N_THREADS > 1024)
    {
        printf("'threads_per_block' cannot be greater than 1024 because of 'multi_dot_kernel'.\n");
        exit (1);
    }

    num_components = *p_num_components;
    totalMM = (*p_totalMM);
    total_grids = (*p_num_components) * (*p_totalMM);
    /* anderson mixing begin if error level becomes less then start_anderson_error */
    start_anderson_error = *p_start_anderson_error;
    /* max number of previous steps to calculate new field when using Anderson mixing */
    max_anderson = *p_max_anderson;
    /* minimum mixing parameter */
    mix_min = *p_mix_min;
    /* initialize mixing parameter */
    mix = *p_init_mix;
    init_mix = *p_init_mix;
    /* number of anderson mixing steps, increases from 0 to max_anderson */
    n_anderson = -1;

    /* record hisotry of wout in GPU device memory */
    cb_wout_hist = new CircularBufferCUDA(max_anderson+1, total_grids);
    /* record hisotry of wout-w in GPU device memory */
    cb_wdiff_hist = new CircularBufferCUDA(max_anderson+1, total_grids);
    /* record hisotry of dot product of wout-w in CPU host memory */
    cb_wdiffdots = new CircularBuffer(max_anderson+1, max_anderson+1);

    /* define arrays for anderson mixing */
    u_nm = new ftsReal*[max_anderson];
    for (int i=0; i<max_anderson; i++)
        u_nm[i] = new ftsReal[max_anderson];
    v_n = new ftsReal[max_anderson];
    a_n = new ftsReal[max_anderson];
    wdiffdots = new ftsReal[max_anderson+1];

    seg = new ftsReal[totalMM];
    cudaMalloc((void**)&seg_d, sizeof(ftsReal)*totalMM);
    temp = new ftsReal[total_grids];
    cudaMalloc((void**)&temp_d, sizeof(ftsReal)*total_grids);

    /* copy segment arrays */
    for(int i=0; i<totalMM; i++)
        seg[i] = p_seg[i];
    cudaMemcpy(seg_d, seg, sizeof(ftsReal)*totalMM,cudaMemcpyHostToDevice);

    /* sum arrays */
    sum = new ftsReal[totalMM];
    cudaMalloc((void**)&sum_d, sizeof(ftsReal)*totalMM);

    /* fields arrays */
	cudaMalloc((void**)&w_diff_d, sizeof(ftsReal)*total_grids);
    cudaMalloc((void**)&w_d, sizeof(ftsReal)*total_grids);

    /* reset_count */
    am_cuda_reset_count_();

}
extern "C" void am_cuda_reset_count_()
{
    /* initialize mixing parameter */
    mix = init_mix;
    /* number of anderson mixing steps, increases from 0 to max_anderson */
    n_anderson = -1;

    cb_wout_hist->reset();
    cb_wdiff_hist->reset();
    cb_wdiffdots->reset();
}
extern "C" void am_cuda_finalize_()
{
    delete cb_wout_hist;
    delete cb_wdiff_hist;
    delete cb_wdiffdots;

    for (int i=0; i<max_anderson; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] wdiffdots;

    delete[] seg;
    cudaFree(seg_d);

    delete[] temp;
    cudaFree(temp_d);

    delete[] sum;
    cudaFree(sum_d);
    
    cudaFree(w_diff_d);
    cudaFree(w_d);
}
extern "C" void am_cuda_caculate_new_fields_(
    ftsReal *w,
    ftsReal *w_out,
    ftsReal *w_diff,
    double *p_old_error_level,
    double *p_error_level)
{
    ftsReal *wout_hist1_d, *wout_hist2_d;
    double old_error_level = *p_old_error_level;
    double error_level = *p_error_level;

    cudaMemcpy(w_diff_d, w_diff, sizeof(ftsReal)*total_grids, cudaMemcpyHostToDevice);

    //printf("mix: %f\n", mix);
    /* condition to start anderson mixing */
    if(error_level < start_anderson_error || n_anderson >= 0)
        n_anderson = n_anderson + 1;
    if( n_anderson >= 0 )
    {
        /* number of histories to use for anderson mixing */
        n_anderson = min(max_anderson, n_anderson);
        /* store the input and output field (the memory is used in a periodic way) */
        cb_wout_hist->insert(w_out);
        cb_wdiff_hist->insert(w_diff);
        /* evaluate wdiff dot products for calculating Unm and Vn in Thompson's paper */
        for(int i=0; i<= n_anderson; i++)
        {
            wdiffdots[i] = multi_dot_cuda(num_components,
                                            w_diff_d, cb_wdiff_hist->getArray(n_anderson-i));
        }
        cb_wdiffdots->insert(wdiffdots);
    }
    /* conditions to apply the simple mixing method */
    if( n_anderson <= 0 )
    {
        /* dynamically change mixing parameter */
        if (old_error_level < error_level)
            mix = max(mix*0.7, mix_min);
        else
            mix = mix*1.01;
        /* make a simple mixing of input and output fields for the next iteration */
        for(int i=0; i<total_grids; i++)
            w[i] = (1.0-mix)*w[i] + mix*w_out[i];
    }
    else
    {
        /* calculate Unm and Vn */
        for(int i=0; i<n_anderson; i++)
        {
            v_n[i] = cb_wdiffdots->getSym(n_anderson, n_anderson)
                         - cb_wdiffdots->getSym(n_anderson, n_anderson-i-1);
            for(int j=0; j<n_anderson; j++)
            {
                u_nm[i][j] = cb_wdiffdots->getSym(n_anderson, n_anderson)
                                      - cb_wdiffdots->getSym(n_anderson, n_anderson-i-1)
                                      - cb_wdiffdots->getSym(n_anderson-j-1, n_anderson)
                                      + cb_wdiffdots->getSym(n_anderson-i-1, n_anderson-j-1);
            }
        }

        find_an(u_nm, v_n, a_n, n_anderson);

        /* calculate the new field */
        wout_hist1_d = cb_wout_hist->getArray(n_anderson);
        cudaMemcpy(w_d, wout_hist1_d, sizeof(ftsReal)*total_grids,cudaMemcpyDeviceToDevice);
        for(int i=0; i<n_anderson; i++)
        {
            wout_hist2_d = cb_wout_hist->getArray(n_anderson-i-1);
            addLinComb<<<N_BLOCKS, N_THREADS>>>(w_d, a_n[i], wout_hist2_d, -a_n[i], wout_hist1_d, total_grids);
        }
        cudaMemcpy(w, w_d, sizeof(ftsReal)*total_grids,cudaMemcpyDeviceToHost);
    }
}
/*
static ftsReal multi_dot(int n_comp, ftsReal *a, ftsReal *b);
static ftsReal multi_dot(int n_comp, ftsReal *a, ftsReal *b)
{
    ftsReal total = 0.0;
    for(int n=0; n<n_comp; n++)
    {
        for(int i=0; i<totalMM; i++)
            total = total + seg[i]*a[i+n*totalMM]*b[i+n*totalMM];
    }
    return total;
}
*/
static ftsReal multi_dot_cuda(int n_comp, ftsReal *a_d, ftsReal *b_d)
{
    ftsReal total = 0.0;

    switch(N_THREADS)
    {
    case 1024:
        multi_dot_kernel<1024><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 512:
        multi_dot_kernel<512><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 256:
        multi_dot_kernel<256><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 128:
        multi_dot_kernel<128><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 64:
        multi_dot_kernel<64><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 32:
        multi_dot_kernel<32><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 16:
        multi_dot_kernel<16><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 8:
        multi_dot_kernel<8><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 4:
        multi_dot_kernel<4><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 2:
        multi_dot_kernel<2><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    case 1:
        multi_dot_kernel<1><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(ftsReal)>>>(n_comp, seg_d, a_d, b_d, sum_d, totalMM);
        break;
    }
    cudaMemcpy(sum, sum_d, sizeof(ftsReal)*N_BLOCKS,cudaMemcpyDeviceToHost);
    for(int i=0; i<N_BLOCKS; i++)
    {
        total += sum[i];
    }
    return total;
}

static void find_an(ftsReal **u, ftsReal *v, ftsReal *a, int n)
{

    int i,j,k;
    ftsReal factor, tempsum;
    /* elimination process */
    for(i=0; i<n; i++)
    {
        for(j=i+1; j<n; j++)
        {
            factor = u[j][i]/u[i][i];
            v[j] = v[j] - v[i]*factor;
            for(k=i+1; k<n; k++)
            {
                u[j][k] = u[j][k] - u[i][k]*factor;
            }
        }
    }
    /* find the solution */
    a[n-1] = v[n-1]/u[n-1][n-1];
    for(i=n-2; i>=0; i--)
    {
        tempsum = 0.0;
        for(j=i+1; j<n; j++)
        {
            tempsum = tempsum + u[i][j]*a[j];
        }
        a[i] = (v[i] - tempsum)/u[i][i];
    }
}
/*
static void print_array(int n, ftsReal *a);
static void print_array(int n, ftsReal *a)
{
for(int i=0; i<n-1; i++)
{
printf("%f, ", a[i]);
}
printf("%f\n", a[n-1]);
}
*/
