/*-------------------------------------------------------------
* This is an abstract SimulationBox class.
* This class defines simulation box parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#ifndef SIMULATION_BOX_H_
#define SIMULATION_BOX_H_

#include <array>
#include <vector>

class SimulationBox
{
protected:
    int dim;        // the dimension of simulation box
    std::array<int,3> nx;  // the number of grid in each direction
    std::array<double,3> lx;  // length of the block copolymer in each direction (in units of aN^1/2)
    std::array<double,3> dx;  // grid interval in each direction
    int n_grid;  // the number of grid
    double *dv; // dV, simple integral weight,
    double volume; // volume of the system.

public:
    SimulationBox(std::vector<int> nx, std::vector<double> lx);
    virtual ~SimulationBox();

    int get_dim();
    int get_nx(int i);
    double get_lx(int i);
    double get_dx(int i);
    std::array<int,3> get_nx();
    std::array<double,3> get_lx();
    std::array<double,3> get_dx();
    double get_dv(int i);
    int get_n_grid();
    double get_volume();

    virtual void set_lx(std::vector<double> new_lx);

    double integral(double *g);
    double inner_product(double *g, double *h);
    double multi_inner_product(int n_comp, double *g, double *h);
    void zero_mean(double *g);

    // Method for SWIG
    double integral(double *g, int len_g)
    {
        return integral(g);
    }
    double inner_product(double *g, int len_g, double *h, int len_h)
    {
        return inner_product(g, h);
    }
    void zero_mean(double *g, int len_g)
    {
        zero_mean(g);
    };
};
#endif
