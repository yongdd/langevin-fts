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

template <typename T>
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

    virtual T integral(const T *g);
    virtual T inner_product(const T *g, const T *h);
    virtual T inner_product_inverse_weight(const T *g, const T *h, const T *w);
    virtual T multi_inner_product(int n_comp, const T *g, const T *h);
    virtual void zero_mean(T *g);

    virtual T integral_device(const T *g)=0;
    virtual T inner_product_device(const T *g, const T *h)=0;
    virtual T inner_product_inverse_weight_device(const T *g, const T *h, const T *w)=0;
    virtual T multi_inner_product_device(int n_comp, const T *g, const T *h)=0;
    virtual void zero_mean_device(T *g)=0;
};
#endif
