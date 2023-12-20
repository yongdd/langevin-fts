
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <map>
#include "ComputationBox.h"

//----------------- Constructor -----------------------------
ComputationBox::ComputationBox(std::vector<int> new_nx, std::vector<double> new_lx, const double* mask)
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
        nx = new_nx;
        lx = new_lx;

        const int DIM = this->dim;

        // Grid interval
        for(int d=0; d<DIM; d++)
            dx.push_back(lx[d]/nx[d]);

        // The number of grids
        n_grid = 1;
        for(int d=0; d<DIM; d++)
            n_grid *= nx[d];

        // Mask
        // Penetrable region == 1.0 
        // Impenetrable region == 0.0
        if (mask != nullptr)
        {
            this->mask = new double[n_grid];
            for(int i=0; i<n_grid; i++)
            {
                if(abs(mask[i]) < 1e-7)
                    this->mask[i] = 0.0;
                else if(abs(mask[i]-1.0) < 1e-7)
                    this->mask[i] = 1.0;
                else
                    throw_with_line_number("mask[" + std::to_string(i) + "] must be 0.0 or 1.0");
            }
        }
        else
            this->mask = nullptr;

        // Weight factor for integral
        dv = new double[n_grid];
        for(int i=0; i<n_grid; i++)
        {
            dv[i] = 1.0;
            for(int d=0; d<DIM; d++)
                dv[i] *= dx[d];
        }
        if (this->mask != nullptr)
            for(int i=0; i<n_grid; i++)
                dv[i] *= this->mask[i];

        // Volume of simulation box
        volume = 1.0;
        for(int d=0; d<DIM; d++)
            volume *= lx[d];

        // Accessible volume
        accessible_volume = 0.0;
        for(int i=0; i<n_grid; i++)
            accessible_volume += dv[i];

        // Set boundary conditions
        // Default is periodic boundary
        for(int i=0; i<2*DIM; i++)
            bc.push_back(BoundaryCondition::PERIODIC);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
ComputationBox::ComputationBox(std::vector<int> new_nx, std::vector<double> new_lx,
    std::vector<std::string> bc, const double* mask)
    : ComputationBox(new_nx, new_lx, mask)
{
    try
    {
        const int DIM = this->dim;
        if(2*DIM != bc.size() && 0 != bc.size())
        {
            throw_with_line_number(
                "We expect 0 or " + std::to_string(2*DIM) + " boundary conditions, but we get " + std::to_string(bc.size()) +
                ". For each dimension, two boundary conditions (top and bottom) are required.");
        }

        if(bc.size() > 0)
        {
            for(int i=0; i<2*DIM; i++)
            {
                std::string bc_name = bc[i];
                // Transform into lower cases
                std::transform(bc_name.begin(), bc_name.end(), bc_name.begin(),
                            [](unsigned char c)
                {
                    return std::tolower(c);
                });

                if(bc_name == "periodic")
                    this->bc[i] = BoundaryCondition::PERIODIC;
                else if(bc_name == "reflecting")
                    this->bc[i] = BoundaryCondition::REFLECTING;
                else if(bc_name == "absorbing")
                    this->bc[i] = BoundaryCondition::ABSORBING;
                else
                    throw_with_line_number(bc_name + " is an invalid boundary condition. Choose among ['periodic', 'reflecting', 'absorbing']");
            }
        }
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
    if (mask != nullptr)
        delete[] mask;
}
//----------------- get methods-------------------------------------
int ComputationBox::get_dim()
{
    return dim;
}
int ComputationBox::get_nx(int i)
{
    if (i < 0 or i >= dim)
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.");
    return nx[i];
}
double ComputationBox::get_lx(int i)
{
    if (i < 0 or i >= dim)
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.");
    return lx[i];
}
double ComputationBox::get_dx(int i)
{
    if (i < 0 or i >= dim)
        throw_with_line_number("'" + std::to_string(i) + "' is out of range.");
    return dx[i];
}
std::vector<int> ComputationBox::get_nx()
{
    return nx;
}
std::vector<double> ComputationBox::get_lx()
{
    return lx;
}
std::vector<double> ComputationBox::get_dx()
{
    return dx;
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
double ComputationBox::get_accessible_volume()
{
    return accessible_volume;
}
const double* ComputationBox::get_mask()
{
    return mask;
}
const std::vector<BoundaryCondition> ComputationBox::get_boundary_conditions()
{
    return bc;
}
BoundaryCondition ComputationBox::get_boundary_condition(int i)
{
    return bc[i];
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

    lx = new_lx;
    // Grid interval
    for(int d=0; d<dim; d++)
        dx[d] = lx[d]/nx[d];

    // Weight factor for integral
    for(int i=0; i<n_grid; i++)
    {
        dv[i] = 1.0;
        for(int d=0; d<dim; d++)
            dv[i] *= dx[d];
    }
    if (this->mask != nullptr)
        for(int i=0; i<n_grid; i++)
            dv[i] *= this->mask[i]; 

    // Volume of simulation box
    volume = 0.0;
    for(int i=0; i<n_grid; i++)
        volume += dv[i];
}
//-----------------------------------------------------------
// This method calculates integral of g
double ComputationBox::integral(const double *g)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i];
    return sum;
}
// This method calculates inner product g and h
double ComputationBox::inner_product(const double *g, const double *h)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i]*h[i];
    return sum;
}
// This method calculates inner product g and h with weight 1/w
double ComputationBox::inner_product_inverse_weight(const double *g, const double *h, const double *w)
{
    double sum{0.0};
    for(int i=0; i<n_grid; i++)
        sum += dv[i]*g[i]*h[i]/w[i];
    return sum;
}
//-----------------------------------------------------------
double ComputationBox::multi_inner_product(int n_comp, const double *g, const double *h)
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
