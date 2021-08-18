
#include "CpuSimulationBox.h"
#include "CpuAndersonMixing.h"

int main()
{
    const int MM{10};
    double dv[MM]{0.0};
    CpuSimulationBox sb({1,2,3},{4,3,2});
    CpuAndersonMixing am(&sb, 2, 10, 0.01, 0.01, 0.1);
}