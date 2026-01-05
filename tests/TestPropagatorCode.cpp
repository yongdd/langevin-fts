#include <iostream>
#include <cmath>
#include <numbers>

#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "Polymer.h"

int main()
{
    try
    {
        std::vector<BlockInput> blocks = {
            {"A",0.4,0,1},
            {"A",1.2,0,2},
            {"B",1.2,0,5},
            {"B",0.9,0,6},
            {"A",0.9,1,4},
            {"A",1.2,1,15},
            {"B",1.2,2,3},
            {"A",0.9,2,7},
            {"B",1.2,2,10},
            {"B",1.2,3,14},
            {"A",0.9,4,8},
            {"A",1.2,4,9},
            {"B",1.2,7,19},
            {"A",0.9,8,13},
            {"B",1.2,9,12},
            {"A",1.2,9,16},
            {"A",1.2,10,11},
            {"B",1.2,13,17},
            {"A",1.2,13,18}};

        Molecules molecules("Continuous", 0.1, {{"A",1.0}, {"B",2.0}});
        molecules.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer propagator_computation_optimizer(&molecules, false);

        // Display all blocks and branches
        propagator_computation_optimizer.display_blocks();
        propagator_computation_optimizer.display_propagators();

        // Test get_deps_from_key
        std::string key;
        std::vector<std::tuple<std::string, int, int>> sub_deps;
        // sub_deps: []
        key = "B";
        sub_deps = PropagatorCode::get_deps_from_key(key);
        if(sub_deps.size() != 0)
            return -1;
        if(PropagatorCode::get_monomer_type_from_key(key) != "B")
            return -1;

        // sub_deps: [A:3, B:3] (using length_index: 3 = 1.2/0.1 = 12 segments)
        key = "(A3B3)A";
        sub_deps = PropagatorCode::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "A" || std::get<1>(sub_deps[0]) != 3)
            return -1;
        if(std::get<0>(sub_deps[1]) != "B" || std::get<1>(sub_deps[1]) != 3)
            return -1;
        if(PropagatorCode::get_monomer_type_from_key(key) != "A")
            return -1;

        // sub_deps: [(A3)B:3, (B3)A:2, (B3)B:3] (length_index: 3=12 segs, 2=9 segs)
        key = "((A3)B3(B3)A2(B3)B3)A";
        sub_deps = PropagatorCode::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "(A3)B" || std::get<1>(sub_deps[0]) != 3)
            return -1;
        if(std::get<0>(sub_deps[1]) != "(B3)A" || std::get<1>(sub_deps[1]) != 2)
            return -1;
       if(std::get<0>(sub_deps[2]) != "(B3)B" || std::get<1>(sub_deps[2]) != 3)
            return -1;
        if(PropagatorCode::get_monomer_type_from_key(key) != "A")
            return -1;

        // sub_deps: [(((((A3B3)A2)A2(A3B3)A3)A2A3)A1B3B2)A:3, (A3)B:3, (B3)B:3]
        // (length_index: 3=12 segs, 2=9 segs, 1=4 segs)
        key = "((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B3B2)A3(A3)B3(B3)B3)A";
        sub_deps = PropagatorCode::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "(((((A3B3)A2)A2(A3B3)A3)A2A3)A1B3B2)A" || std::get<1>(sub_deps[0]) != 3)
            return -1;
        if(std::get<0>(sub_deps[1]) != "(A3)B" || std::get<1>(sub_deps[1]) != 3)
            return -1;
        if(std::get<0>(sub_deps[2]) != "(B3)B" || std::get<1>(sub_deps[2]) != 3)
            return -1;
        if(PropagatorCode::get_monomer_type_from_key(key) != "A")
            return -1;

        // Print sub propagator_codes
        std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators = propagator_computation_optimizer.get_computation_propagators();
        for(const auto& item : computation_propagators)
        {
            std::cout << item.first << ":\n\t";
            std::cout << "{max_n_segment: " << item.second.max_n_segment << ",\n\tsub_deps: [";
            sub_deps = PropagatorCode::get_deps_from_key(item.first);
            for(size_t i=0; i<sub_deps.size(); i++)
                std::cout << std::get<0>(sub_deps[i]) << ":" << std::get<1>(sub_deps[i]) << ", " ;
            std::cout << "]}" << std::endl;
        }

        // Check size of sub_deps dictionary
        if(propagator_computation_optimizer.get_n_computation_propagator_codes() != 29)
            return -1;

        // Check max_n_segment (keys now use length_index: 3=12 segs, 2=9 segs, 1=4 segs)
        if(propagator_computation_optimizer.get_computation_propagator("B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(B3)B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(B3)A").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(A3B3)A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(A3)B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((A3B3)A2)A").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((A3)B3(B3)A2(B3)B3)A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((A3B3)A2)A2(A3B3)A3)A").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((A3)B3(B3)A2(B3)B3)A3B2B3)A").max_n_segment != 4)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((A3B3)A2)A2(A3B3)A3)A2A3)A").max_n_segment != 4)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((A3)B3(B3)A2(B3)B3)A3B2B3)A1(((A3B3)A2)A2(A3B3)A3)A2)A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3B3)A2)A2(A3B3)A3)A2A3)A1((A3)B3(B3)A2(B3)B3)A3B2)B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3B3)A2)A2(A3B3)A3)A2A3)A1((A3)B3(B3)A2(B3)B3)A3B3)B").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2(A3B3)A3)A").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2((A3B3)A2)A2)A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(B3)A2(B3)B3)B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)B3)A").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)A2)B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2(A3B3)A3)A2)A").max_n_segment != 9)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2((A3B3)A2)A2)A3B3)A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2((A3B3)A2)A2)A3A3)B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(B3)A2(B3)B3)B3)A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)B3)A2)B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3B3)A2)A2(A3B3)A3)A2A3)A1B2B3)A3(A3)B3(B3)A2)B3)B").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2(A3B3)A3)A2)A2B3)A").max_n_segment != 12)
            return -1;
        if(propagator_computation_optimizer.get_computation_propagator("(((((((A3)B3(B3)A2(B3)B3)A3B2B3)A1A3)A2(A3B3)A3)A2)A2A3)B").max_n_segment != 12)
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
