
#include "MklAndersonMixing.h"

int main()
{
    const int MM{10};
    double dv[MM]{0.0};
    AnderosnMixing am(2, MM, dv, 10, 0.01, 0.01, 0.1);
}
