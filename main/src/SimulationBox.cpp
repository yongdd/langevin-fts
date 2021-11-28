
#include <iostream>
#include "SimulationBox.h"

//----------------- Constructor -----------------------------
SimulationBox::SimulationBox(std::vector<int> new_nx, std::vector<double> new_lx)
{
    if ( new_nx.size() != new_lx.size() )
    {
        std::cerr << "The sizes of nx and lx are not the same. " << std::endl;
        exit(-1);
    }
    this->dim = new_nx.size();
    if ( dim != 3 && dim != 2 && dim != 1 )
    {
        std::cerr << "We expect 1D, 2D or 3D, but we get " << dim <<std::endl;
        exit(-1);
    }

    for(int i=0; i<dim; i++)
    {
        nx[i] = new_nx[i];
        lx[i] = new_lx[i];
        dx[i] = new_lx[i]/new_nx[i];
    }
    if (dim == 2 )
    {
        nx[2] = 1;
        lx[2] = 1.0;
        dx[2] = 1.0;
    }
    else if (dim == 1 )
    {
        nx[1] = 1;
        lx[1] = 1.0;
        dx[1] = 1.0;

        nx[2] = 1;
        lx[2] = 1.0;
        dx[2] = 1.0;
    }
    // the number of grids
    n_grid = nx[0]*nx[1]*nx[2];
    // weight factor for integral
    dv = new double[n_grid];
    for(int i=0; i<n_grid; i++)
        dv[i] = dx[0]*dx[1]*dx[2];
    // system polymer
    volume = lx[0]*lx[1]*lx[2];
}
//----------------- Destructor -----------------------------
SimulationBox::~SimulationBox()
{
    delete[] dv;
}

double dv_at(int i);
//----------------- get methods-------------------------------------
int SimulationBox::get_dim()
{
    return dim;
}
int SimulationBox::get_nx(int i)
{
    return nx[i];
}
double SimulationBox::get_lx(int i)
{
    return lx[i];
}
double SimulationBox::get_dx(int i)
{
    return dx[i];
}
std::array<int,3> SimulationBox::get_nx()
{
    return {nx[0],nx[1],nx[2]};
}
std::array<double,3> SimulationBox::get_lx()
{
    return {lx[0],lx[1],lx[2]};
}
std::array<double,3> SimulationBox::get_dx()
{
    return {dx[0],dx[1],dx[2]};
}
double SimulationBox::get_dv(int i)
{
    return dv[i];
}
int SimulationBox::get_n_grid()
{
    return n_grid;
}
double SimulationBox::get_volume()
{
    return volume;
}
//-----------------------------------------------------------
// This method calculates inner product g and h
double SimulationBox::integral(double *g)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i];
    return sum;
}
// This method calculates inner product g and h
double SimulationBox::inner_product(double *g, double *h)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i]*h[i];
    return sum;
}
//-----------------------------------------------------------
double SimulationBox::multi_inner_product(int n_comp, double *g, double *h)
{
    double sum{0.0};
    for(int n=0; n < n_comp; n++)
    {
        for(int i=0; i<n_grid; i++)
            sum += dv[i]*g[i+n*n_grid]*h[i+n*n_grid];
    }
    return sum;
}
//-----------------------------------------------------------
// This method makes the input a zero-meaned matrix
void SimulationBox::zero_mean(double *g)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i];
    for(int i=0; i<n_grid; i++)
        g[i] -= sum/volume;
}
