
#include <iostream>
#include <sstream>
#include "ComputationBox.h"

//----------------- Constructor -----------------------------
ComputationBox::ComputationBox(std::vector<int> new_nx, std::vector<double> new_lx)
{
    if ( new_nx.size() != new_lx.size() )
        throw_with_line_number("The sizes of nx (" + std::to_string(new_nx.size()) + ") and lx (" + std::to_string(new_lx.size()) + ") must match.");
    if ( new_nx.size() != 3 && new_nx.size() != 2 && new_nx.size() != 1)
        throw_with_line_number("We expect 1D, 2D or 3D, but we get " + std::to_string(new_nx.size()));
    if (std::any_of(new_nx.begin(), new_nx.end(), [](int nx) { return nx <= 0;})){
        std::stringstream ss_nx;
        std::copy(new_nx.begin(), new_nx.end(), std::ostream_iterator<int>(ss_nx, ", "));
        std::string str_nx = ss_nx.str();
        str_nx = str_nx.substr(0, str_nx.length()-2);
        throw_with_line_number("nx (" + str_nx + ") must be positive numbers");
    }
    if (std::any_of(new_lx.begin(), new_lx.end(), [](double lx) { return lx <= 0.0;})){
        std::stringstream ss_lx;
        std::copy(new_lx.begin(), new_lx.end(), std::ostream_iterator<int>(ss_lx, ", "));
        std::string str_lx = ss_lx.str();
        str_lx = str_lx.substr(0, str_lx.length()-2);
        throw_with_line_number("lx (" + str_lx + ") must be positive numbers");
    }

    try
    {
        this->dim = new_nx.size();
        for(int i=0; i<dim; i++)
        {
            nx[i+3-dim] = new_nx[i];
            lx[i+3-dim] = new_lx[i];
            dx[i+3-dim] = new_lx[i]/new_nx[i];
        }
        if (dim == 2 )
        {
            nx[0] = 1;
            lx[0] = 1.0;
            dx[0] = 1.0;
        }
        else if (dim == 1 )
        {
            nx[0] = 1;
            lx[0] = 1.0;
            dx[0] = 1.0;

            nx[1] = 1;
            lx[1] = 1.0;
            dx[1] = 1.0;
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
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
//----------------- Destructor -----------------------------
ComputationBox::~ComputationBox()
{
    delete[] dv;
}

double dv_at(int i);
//----------------- get methods-------------------------------------
int ComputationBox::get_dim()
{
    return dim;
}
int ComputationBox::get_nx(int i)
{
    return nx[i];
}
double ComputationBox::get_lx(int i)
{
    return lx[i];
}
double ComputationBox::get_dx(int i)
{
    return dx[i];
}
std::array<int,3> ComputationBox::get_nx()
{
    return {nx[0],nx[1],nx[2]};
}
std::array<double,3> ComputationBox::get_lx()
{
    return {lx[0],lx[1],lx[2]};
}
std::array<double,3> ComputationBox::get_dx()
{
    return {dx[0],dx[1],dx[2]};
}
double ComputationBox::get_dv(int i)
{
    return dv[i];
}
int ComputationBox::get_n_grid()
{
    return n_grid;
}
double ComputationBox::get_volume()
{
    return volume;
}
//----------------- set methods-------------------------------------
void ComputationBox::set_lx(std::vector<double> new_lx)
{
    if ( new_lx.size() != (unsigned int) dim )
        throw_with_line_number("The sizes of new lx (" + std::to_string(new_lx.size()) + ") and dim (" + std::to_string(dim) + ") must match.");

    if (std::any_of(new_lx.begin(), new_lx.end(), [](double lx) { return lx <= 0.0;})){
        std::stringstream ss_lx;
        std::copy(new_lx.begin(), new_lx.end(), std::ostream_iterator<int>(ss_lx, ", "));
        std::string str_lx = ss_lx.str();
        str_lx = str_lx.substr(0, str_lx.length()-2);
        throw_with_line_number("new lx (" + str_lx + ") must be positive numbers");
    }

    for(int i=0; i<dim; i++)
    {
        lx[i+3-dim] = new_lx[i];
        dx[i+3-dim] = new_lx[i]/nx[i+3-dim];
    }
    if (dim == 2 )
    {
        lx[0] = 1.0;
        dx[0] = 1.0;
    }
    else if (dim == 1 )
    {
        lx[0] = 1.0;
        dx[0] = 1.0;

        lx[1] = 1.0;
        dx[1] = 1.0;
    }
    // weight factor for integral
    for(int i=0; i<n_grid; i++)
        dv[i] = dx[0]*dx[1]*dx[2];
    volume = lx[0]*lx[1]*lx[2];
}
//-----------------------------------------------------------
// This method calculates integral of g
double ComputationBox::integral(double *g)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i];
    return sum;
}
// This method calculates inner product g and h
double ComputationBox::inner_product(double *g, double *h)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i]*h[i];
    return sum;
}

// This method calculates inner product g and h with weight 1/w
double ComputationBox::inner_product_inverse_weight(double *g, double *h, double *w)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i]*h[i]/w[i];
    return sum;
}
//-----------------------------------------------------------
double ComputationBox::multi_inner_product(int n_comp, double *g, double *h)
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
void ComputationBox::zero_mean(double *g)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i];
    for(int i=0; i<n_grid; i++)
        g[i] -= sum/volume;
}
