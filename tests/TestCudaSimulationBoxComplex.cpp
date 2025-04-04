#include <iostream>
#include <cmath>
#include <complex>

#include "CudaComputationBox.h"

int main()
{
    try
    {
        CudaComputationBox<std::complex<double>>  cb({1,2,2}, {4.5,3.5,2.5}, {});
        double volume;
        std::complex<double> sum_w;
        std::complex<double> w[] 
            {{-0.8632002499891152, -0.2642949058151489},
            {-0.4400105438417454, 0.25515839348886193},
            {0.11957624566912606, -0.47053949095489656},
            {-0.47288841841362395, -0.989970480481541}};

        ftsComplex g[]
            {{-0.9415024129393723, -0.49812524902468414},
            {0.41847784612200645, 0.8890323006002832},
            {-0.36140888131449245, -0.029759426767947073},
            {0.39470228341808133, -0.7504819613930107}};

        ftsComplex h[] 
            {{0.32138498226345313, -0.030002215677094046},
            {0.5460116832979387, 0.12872199923910688},
            {0.4397920549860139, 0.44860363056686503},
            {0.6490069678310173, -0.6164591864981155}};

        ftsComplex gm[] 
            {{-0.7597619303835041, 0.10250398897632951},
            {-0.034879315428214985, 0.6421559379075319},
            {0.6520770931658473, 0.0947148787578771},
            {0.1147344011203335, -0.0013486534653663895},
            {0.6665403162481605, 0.14194066509526593},
            {0.2521801134963888, -0.6413889885242658},
            {0.9808898058112421, -0.6708860065130271},
            {0.509802309538298, -0.14069459509112625}};
        ftsComplex hm[]
            {{-0.23168234399177323, 0.17795759353248086},
            {0.8534783611236558, -0.6583778325855023},
            {0.7255420705159961, -0.9590076481309235},
            {0.35090074919266856, 0.3540832912909959},
            {-0.8946471983599762, -0.7748500391040254},
            {-0.2910449481410968, 0.7333947043702351},
            {0.8981178670015273, -0.7416090497144214},
            {-0.3576265073984164, 0.6277381773607089}};

        ftsComplex *d_g, *d_h, *d_gm, *d_hm;
        cudaMalloc((void**)&d_g, sizeof(ftsComplex) * cb.get_total_grid());
        cudaMalloc((void**)&d_h, sizeof(ftsComplex) * cb.get_total_grid());
        cudaMalloc((void**)&d_gm, sizeof(ftsComplex) * cb.get_total_grid() * 2);
        cudaMalloc((void**)&d_hm, sizeof(ftsComplex) * cb.get_total_grid() * 2);
        cudaMemcpy(d_g, g, sizeof(ftsComplex) * cb.get_total_grid(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_h, h, sizeof(ftsComplex) * cb.get_total_grid(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gm, gm, sizeof(ftsComplex) * cb.get_total_grid() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_hm, hm, sizeof(ftsComplex) * cb.get_total_grid() * 2, cudaMemcpyHostToDevice);

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

        std::cout<< "sum(dV), Volume: " << volume <<" "<< cb.get_volume() << std::endl;
        std::cout<< "inner product: "<< cb.inner_product_device(d_g,d_h) << std::endl;
        std::cout<< "multi inner product: "<< cb.multi_inner_product_device(2, d_gm,d_hm) << std::endl;
        std::cout<< "sum(w): " << sum_w << std::endl;
        
        if( std::abs(cb.inner_product_device(d_g,d_h)-std::complex<double>{-5.468649581714552, -4.903729566482727}) > 1e-5)
            return -1;
        if( std::abs(cb.multi_inner_product_device(2,d_gm,d_hm)-std::complex<double>{13.343735222259584, -13.14993884723823}) > 1e-5)
            return -1;
        if( std::abs(sum_w) > 1e-7)
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
