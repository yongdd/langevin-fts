/*-------------------------------------------------------------
* This is an abstract ComputationBox class.
* This class defines computation grids and Lengths parameters, and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#ifndef COMPUTATION_BOX_H_
#define COMPUTATION_BOX_H_

#include <array>
#include <vector>
#include <cassert>

#include "Exception.h"

enum class BoundaryCondition
{
	PERIODIC,
	REFLECTING,
	ABSORBING,
};

class ComputationBox
{
protected:
    int dim;        // the dimension of Computation Grids and Lengths
    std::vector<int> nx;  // the number of grid in each direction
    std::vector<double> lx;  // length of the block copolymer in each direction (in units of aN^1/2)
    std::vector<double> dx;  // grid interval in each direction
    int total_grid;  // the number of grid
    double *mask; // mask, impenetrable region
    double *dv; // dV, simple integral weight,
    double volume; // volume of the system.
    double accessible_volume; // accessible volume excluding mask region
    
    // boundary conditions 
    // "xl": bc at x = 0
    // "xh": bc at x = Lx
    // "yl": bc at y = 0
    // "yh": bc at y = Ly
    // "zl": bc at z = 0
    // "zh": bc at z = Lz
    std::vector<BoundaryCondition> bc;

public:
    ComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr);
    virtual ~ComputationBox();

    int get_dim();
    std::vector<int> get_nx();
    int get_nx(int i);
    std::vector<double> get_lx();
    double get_lx(int i);
    std::vector<double> get_dx();
    double get_dx(int i);
    double get_dv(int i);
    int get_total_grid();
    double get_volume();
    double get_accessible_volume();
    const double* get_mask() ;
    const std::vector<BoundaryCondition> get_boundary_conditions();
    BoundaryCondition get_boundary_condition(int i);

    virtual void set_lx(std::vector<double> new_lx);

    virtual double integral(const double *g);
    virtual double inner_product(const double *g, const double *h);
    virtual double inner_product_inverse_weight(const double *g, const double *h, const double *w);
    virtual double multi_inner_product(int n_comp, const double *g, const double *h);
    virtual void zero_mean(double *g);

    virtual double integral_device(const double *g)=0;
    virtual double inner_product_device(const double *g, const double *h)=0;
    virtual double inner_product_inverse_weight_device(const double *g, const double *h, const double *w)=0;
    virtual double multi_inner_product_device(int n_comp, const double *g, const double *h)=0;
    virtual void zero_mean_device(double *g)=0;
};
#endif
