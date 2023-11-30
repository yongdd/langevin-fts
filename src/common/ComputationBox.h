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

class ComputationBox
{
protected:
    int dim;        // the dimension of Computation Grids and Lengths
    std::vector<int> nx;  // the number of grid in each direction
    std::vector<double> lx;  // length of the block copolymer in each direction (in units of aN^1/2)
    std::vector<double> dx;  // grid interval in each direction
    int n_grid;  // the number of grid
    double *dv; // dV, simple integral weight,
    double volume; // volume of the system.

public:

    ComputationBox(std::vector<int> nx, std::vector<double> lx);
    virtual ~ComputationBox();

    int get_dim();
    std::vector<int> get_nx();
    int get_nx(int i);
    std::vector<double> get_lx();
    double get_lx(int i);
    std::vector<double> get_dx();
    double get_dx(int i);
    double get_dv(int i);
    int get_n_grid();
    double get_volume();

    virtual void set_lx(std::vector<double> new_lx);

    virtual double integral(double *g);
    virtual double inner_product(double *g, double *h);
    virtual double inner_product_inverse_weight(double *g, double *h, double *w);
    virtual double multi_inner_product(int n_comp, double *g, double *h);
    virtual void zero_mean(double *g);

    virtual double integral_device(double *g)=0;
    virtual double inner_product_device(double *g, double *h)=0;
    virtual double inner_product_inverse_weight_device(double *g, double *h, double *w)=0;
    virtual double multi_inner_product_device(int n_comp, double *g, double *h)=0;
    virtual void zero_mean_device(double *g)=0;
};
#endif
