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

    int nx[3];  // the number of grid in each direction
    int MM;  // the number of total grid
    double lx[3];  // length of the block copolymer in each direction (in units of aN^1/2)
    double dx[3];  // grid interval in each direction
    double *dv; // dV, simple integral weight,
    double volume; // volume of the system.

    SimulationBox(
        std::array<int,3> nx, std::array<double,3> lx);
    SimulationBox(int *nx, double *lx)
     : SimulationBox({nx[0],nx[1],nx[2]},{lx[0],lx[1],lx[2]}) {};
    virtual ~SimulationBox();

    double dot(double *g, double *h);
    double multi_dot(int n_comp, double *g, double *h);
    void zero_mean(double *w);
};
#endif
