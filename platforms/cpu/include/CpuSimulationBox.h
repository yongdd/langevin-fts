/*-------------------------------------------------------------
* This is a derived CpuSimulationBox class
*--------------------------------------------------------------*/
#ifndef CPU_SIMULATION_BOX_H_
#define CPU_SIMULATION_BOX_H_

#include <array>
#include "SimulationBox.h"

class CpuSimulationBox : public SimulationBox
{
private:
public:
    CpuSimulationBox(std::array<int,3> nx, std::array<double,3> lx);
    CpuSimulationBox(int *nx, double *lx) : CpuSimulationBox({nx[0],nx[1],nx[2]}, {lx[0],lx[1],lx[2]}) {};
};
#endif
