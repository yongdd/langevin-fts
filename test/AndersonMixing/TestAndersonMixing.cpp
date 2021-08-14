
#include "SimulationBox.h"
#include "AndersonMixing.h"

int main()
{
    const int MM{10};
    double dv[MM]{0.0};
    SimulationBox sb({1,2,3},{4,3,2});
    AnderosnMixing am(&sb, 2, 10, 0.01, 0.01, 0.1);
}
