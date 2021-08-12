#include "SimulationBox.h"

int main()
{
    SimulationBox sb({1,1,3}, {4.5,3.5,2.5});

    std::cout<< "Nx: "  << sb.nx[0] <<" "<< sb.nx[1] <<" "<< sb.nx[2] << std::endl;
    std::cout<< "Lx: " << sb.lx[0] <<" "<< sb.lx[1] <<" "<< sb.lx[2] << std::endl;
    std::cout<< "dx: " << sb.dx[0] <<" "<< sb.dx[1] <<" "<< sb.dx[2] << std::endl;

    double sum{0.0};
    for(int i=0; i<sb.total_grids; i++)
    {
        sum += sb.dv[i];
    }
    std::cout<< "sum(dV), Volume: " << sum <<" "<< sb.volume << std::endl;

    double g[] {1.2, 3.4, 2.5};
    double h[] {3.1, -1.4, -2.5};
    std::cout<< "dot : "<< sb.dot(g,h) << std::endl;

    sb.zeromean(g);
    std::cout<< "zeromean g: ";
    for(int i=0; i<sb.total_grids; i++)
    {
        std::cout<< g[i] << " ";
    }
    std::cout<< std::endl;
    return 0;
}
