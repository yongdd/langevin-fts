#include <iostream>
#include <cmath>

#include "Molecules.h"
#include "Scheduler.h"

int main()
{
    try
    {
        Molecules molecules("Continuous", 0.1, {{"A",1.0}, {"B",2.0}}, false);
        molecules.add_polymer(1.0, 
            {"A","A","B","B","A","A","B","A","B","B","A","A","B","A","B","A","A","B","A"},
            {0.4,1.2,1.2,0.9,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,0.9,0.9},
            {0,0,0,0,1,1,2,2,2,3,4,4,7,8,9,9,10,13,13},
            {1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18}, {});

        Scheduler sc(molecules.get_essential_propagator_codes(), 4);

        sc.display(molecules.get_essential_propagator_codes());

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
