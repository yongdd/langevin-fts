#include <iostream>
#include <cmath>

#include "Molecules.h"
#include "PropagatorAnalyzer.h"
#include "Scheduler.h"

int main()
{
    try
    {
        // std::vector<std::string> block_monomer_types = {"A","A","B","B","A","A","B","A","B","B","A","A","B","A","B","A","A","B","A"};
        // std::vector<double> contour_lengths = {0.4,1.2,1.2,0.9,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,0.9,0.9};
        // std::vector<int> v = {0,0,0,0,1,1,2,2,2,3,4,4,7,8,9,9,10,13,13};
        // std::vector<int> u = {1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18};

        // for(size_t p=0; p<block_monomer_types.size(); p++){
        //     printf("{\"%s\",%4.1f,%2d,%2d},\n", block_monomer_types[p].c_str(), contour_lengths[p], v[p], u[p]);
        // }

        std::vector<BlockInput> blocks =
        {
            {"A", 0.4, 0, 1},
            {"A", 1.2, 0, 2},
            {"B", 1.2, 0, 5},
            {"B", 0.9, 0, 6},
            {"A", 0.9, 1, 4},
            {"A", 1.2, 1,15},
            {"B", 1.2, 2, 3},
            {"A", 0.9, 2, 7},
            {"B", 1.2, 2,10},
            {"B", 1.2, 3,14},
            {"A", 0.9, 4, 8},
            {"A", 1.2, 4, 9},
            {"B", 1.2, 7,19},
            {"A", 0.9, 8,13},
            {"B", 1.2, 9,12},
            {"A", 1.2, 9,16},
            {"A", 1.2,10,11},
            {"B", 0.9,13,17},
            {"A", 0.9,13,18},
        };

        Molecules molecules("Continuous", 0.1, {{"A",1.0}, {"B",2.0}});
        molecules.add_polymer(1.0, blocks, {});

        PropagatorAnalyzer propagator_analyzer(&molecules, false);

        Scheduler sc(propagator_analyzer.get_computation_propagators(), 4);

        sc.display(propagator_analyzer.get_computation_propagators());

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
