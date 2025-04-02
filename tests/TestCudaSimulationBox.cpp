#include <iostream>
#include <cmath>
#include "CudaComputationBox.h"

int main()
{
    try
    {
        CudaComputationBox cb({1,2,2}, {4.5,3.5,2.5}, {});
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
        
        if( std::abs(cb.inner_product(g,h)-81.553611) > 1e-5)
            return -1;
        if( std::abs(cb.multi_inner_product(2,gm,hm)+14.391321) > 1e-5)
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
