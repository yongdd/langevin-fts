/*-------------------------------------------------------------
* This is an abstract SimulationBox class.
* This class defines simulation box parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#ifndef SIMULATION_BOX_H_
#define SIMULATION_BOX_H_

#include <array>

class SimulationBox
{
private:
public:

    std::array<int,3> nx;  // the number of grid in each direction
    std::array<double,3> lx;  // length of the block copolymer in each direction (in units of aN^1/2)
    std::array<double,3> dx;  // grid interval in each direction
    int MM;  // the number of total grid
    double *dv; // dV, simple integral weight,
    double volume; // volume of the system.

    SimulationBox(
        std::array<int,3> nx, std::array<double,3> lx);
    SimulationBox(int *nx, double *lx)
        : SimulationBox({nx[0],nx[1],nx[2]},
    {
        lx[0],lx[1],lx[2]
    }) {};
    virtual ~SimulationBox();

    double dv_at(int i);
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