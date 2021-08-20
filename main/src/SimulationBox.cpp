#include "SimulationBox.h"

//----------------- Constructor -----------------------------
SimulationBox::SimulationBox(std::array<int,3> nx, std::array<double,3> lx)
{
    for(int i=0; i<3; i++)
    {
        this->nx[i] = nx[i];
        this->lx[i] = lx[i];
        this->dx[i] = lx[i]/nx[i];
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
double SimulationBox::get_dv(int i)
{
    return dv[i];
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
