/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

#include <iostream>
#include <complex>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>
#include "CudaComputationBox.h"
#include "SpaceGroup.h"
#include "TestNanSafe.h"

// Device reductions of a box against its host ones, for two fields of length n
// (the full grid, or the reduced basis with a space group); returns the largest
// difference, including the field after zero_mean_device
static double device_host_difference(CudaComputationBox<double>& cb, int n)
{
    std::vector<double> g(2*n), h(2*n), g_device(n);
    for (int i = 0; i < 2*n; i++)
    {
        g[i] = std::sin(0.7*i + 0.3) + 0.5;
        h[i] = std::cos(1.3*i) - 0.2;
    }
    double *d_g, *d_h;
    CUDA_CHECK(cudaMalloc((void**)&d_g, sizeof(double)*2*n));
    CUDA_CHECK(cudaMalloc((void**)&d_h, sizeof(double)*2*n));
    CUDA_CHECK(cudaMemcpy(d_g, g.data(), sizeof(double)*2*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h, h.data(), sizeof(double)*2*n, cudaMemcpyHostToDevice));

    double diff = std::abs(cb.integral_device(d_g) - cb.integral(g.data()));
    diff = max_or_nan(diff, std::abs(cb.inner_product_device(d_g, d_h) - cb.inner_product(g.data(), h.data())));
    diff = max_or_nan(diff, std::abs(cb.multi_inner_product_device(2, d_g, d_h) - cb.multi_inner_product(2, g.data(), h.data())));

    cb.zero_mean(g.data());
    cb.zero_mean_device(d_g);
    CUDA_CHECK(cudaMemcpy(g_device.data(), d_g, sizeof(double)*n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++)
        diff = max_or_nan(diff, std::abs(g_device[i] - g[i]));

    cudaFree(d_g);
    cudaFree(d_h);
    return diff;
}

int main()
{
    try
    {
        CudaComputationBox<double> cb({1,2,2}, {4.5,3.5,2.5}, {});
        double sum_w, volume;
        double w[] {-0.81747, -2.02147, -3.52469, 1.43482};
        double g[] {2.82006, -0.944886, -1.37078, -2.47236};
        double h[] {4.40394, 1.71801, 2.73214, -0.499087};
        double gm[] {1.21997, 0.130716, -4.524, -1.09946, 1.79612, 0.367613, -0.670658, 1.53048};
        double hm[] {0.443984, 1.55078, 3.26028, -1.07704, 1.37609, -2.33879, -4.88014, 4.23021};

        volume = 0.0;
        for(int i=0; i<cb.get_total_grid(); i++)
            volume += cb.get_dv(i);
        
        cb.zero_mean(w);
        sum_w = 0.0;
        for(int i=0; i<cb.get_total_grid(); i++)
            sum_w += w[i];
        
        std::cout<< "nx: " << cb.get_nx(0) <<" "<< cb.get_nx(1) <<" "<< cb.get_nx(2) << std::endl;
        std::cout<< "lx: " << cb.get_lx(0) <<" "<< cb.get_lx(1) <<" "<< cb.get_lx(2) << std::endl;
        std::cout<< "dx: " << cb.get_dx(0) <<" "<< cb.get_dx(1) <<" "<< cb.get_dx(2) << std::endl;

        //std::cout<< "sum(dV), Volume: " << volume <<" "<< cb.volume << std::endl;
        std::cout<< "inner product: "<< cb.inner_product(g,h) << std::endl;
        std::cout<< "multi inner product: "<< cb.multi_inner_product(2, gm,hm) << std::endl;
        std::cout<< "sum(w): " << sum_w << std::endl;
        
        if( !(std::abs(cb.inner_product(g,h)-81.553611) <= 1e-5))
            return -1;
        if( !(std::abs(cb.multi_inner_product(2,gm,hm)+14.391321) <= 1e-5))
            return -1;
        if( !(std::abs(sum_w) <= 1e-7))
            return -1;

        // Device reductions = host reductions, on the full grid and on the reduced
        // basis of Im-3m with a mask of that symmetry (cavities at the corner and the
        // body center). Until 2026-10 zero_mean_device added the sum to real fields
        // instead of subtracting the mean, and the device multi inner product and
        // zero mean ignored the space group.
        double diff_full = device_host_difference(cb, cb.get_total_grid());
        std::cout<< "device - host reductions, full grid: " << diff_full << std::endl;
        if( !(diff_full <= 1e-10))
            return -1;

        const int n = 16;
        std::vector<double> mask(n*n*n, 1.0);
        for(int i=0; i<n*n*n; i++)
        {
            const int idx[3] {i/(n*n), (i/n)%n, i%n};
            for(double c : {0.0, 0.5})
            {
                double dist2 = 0.0;
                for(int d=0; d<3; d++)
                {
                    double x = std::abs((idx[d] + 0.5)/n - c);
                    x = std::min(x, 1.0 - x);
                    dist2 += x*x;
                }
                if (dist2 < 0.15*0.15)
                    mask[i] = 0.0;
            }
        }
        CudaComputationBox<double> cb_sg({n,n,n}, {3.0,3.0,3.0}, {}, mask.data());
        SpaceGroup sg({n,n,n}, "Im-3m", 529);
        cb_sg.set_space_group(&sg);
        double diff_sg = device_host_difference(cb_sg, cb_sg.get_n_basis());
        std::cout<< "device - host reductions, Im-3m reduced basis with a mask: " << diff_sg << std::endl;
        if( !(diff_sg <= 1e-10))
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
