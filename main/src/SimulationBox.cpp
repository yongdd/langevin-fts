
#include <iostream>
#include "SimulationBox.h"

//----------------- Constructor -----------------------------
SimulationBox::SimulationBox(std::vector<int> new_nx, std::vector<double> new_lx)
{
    if ( new_nx.size() != new_lx.size() ){ 
        std::cerr << "The sizes of nx and lx are not the same. " << std::endl;
        exit(-1);
    }
    this->dimension = new_nx.size();
    if ( dimension != 3 && dimension != 2 ){ 
        std::cerr << "We expect 2D or 3D, but we get " << dimension <<std::endl;
        exit(-1);
    }
    
    for(int i=0; i<dimension; i++)
    {
        nx[i] = new_nx[i];
        lx[i] = new_lx[i];
        dx[i] = new_lx[i]/new_nx[i];
    }
    if (dimension ==2 ){
        nx[2] = 1;
        lx[2] = 1.0;
        dx[2] = 1.0;
    }
    
    // the number of total grids
    MM = nx[0]*nx[1]*nx[2];
    // weight factor for integral
    dv = new double[MM];
    for(int i=0; i<MM; i++)
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
int SimulationBox::get_dimension()
{
    return dimension;
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
int SimulationBox::get_MM()
{
    return MM;
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
    for(int i=0; i<MM; i++)
        sum += dv[i]*g[i];
    return sum;
}
// This method calculates inner product g and h
double SimulationBox::inner_product(double *g, double *h)
{
    double sum{0.0};
    for(int i=0; i<MM; i++)
        sum += dv[i]*g[i]*h[i];
    return sum;
}
//-----------------------------------------------------------
double SimulationBox::multi_inner_product(int n_comp, double *g, double *h)
{
    double sum{0.0};
    for(int n=0; n<n_comp; n++)
    {
        for(int i=0; i<MM; i++)
            sum += dv[i]*g[i+n*MM]*h[i+n*MM];
    }
    return sum;
}
//-----------------------------------------------------------
// This method makes the input a zero-meaned matrix
void SimulationBox::zero_mean(double *g)
{
    double sum{0.0};
    for(int i=0; i<MM; i++)
        sum += dv[i]*g[i];
    for(int i=0; i<MM; i++)
        g[i] -= sum/volume;
}
