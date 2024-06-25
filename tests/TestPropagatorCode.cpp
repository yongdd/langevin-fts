#include <iostream>
#include <cmath>

#include "Molecules.h"
#include "PropagatorAnalyzer.h"
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
        PropagatorAnalyzer propagator_analyzer(&molecules, false);

        // Display all blocks and branches
        propagator_analyzer.display_blocks();
        propagator_analyzer.display_propagators();

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

        // sub_deps: [A:12, B:12]
        key = "(A12B12)A";
        sub_deps = PropagatorCode::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "A" || std::get<1>(sub_deps[0]) != 12)
            return -1;
        if(std::get<0>(sub_deps[1]) != "B" || std::get<1>(sub_deps[1]) != 12)
            return -1;
        if(PropagatorCode::get_monomer_type_from_key(key) != "A")
            return -1;

        // sub_deps: [(A12)B:12, (B12)A:9, (B12)B:12]
        key = "((A12)B12(B12)A9(B12)B12)A";
        sub_deps = PropagatorCode::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "(A12)B" || std::get<1>(sub_deps[0]) != 12)
            return -1;
        if(std::get<0>(sub_deps[1]) != "(B12)A" || std::get<1>(sub_deps[1]) != 9)
            return -1;
       if(std::get<0>(sub_deps[2]) != "(B12)B" || std::get<1>(sub_deps[2]) != 12)
            return -1;
        if(PropagatorCode::get_monomer_type_from_key(key) != "A")
            return -1;

        // sub_deps: [(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A:12, (A12)B:12, (B12)B:12]}
        key = "((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A";
        sub_deps = PropagatorCode::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A" || std::get<1>(sub_deps[0]) != 12)
            return -1;
        if(std::get<0>(sub_deps[1]) != "(A12)B" || std::get<1>(sub_deps[1]) != 12)
            return -1;
        if(std::get<0>(sub_deps[2]) != "(B12)B" || std::get<1>(sub_deps[2]) != 12)
            return -1;
        if(PropagatorCode::get_monomer_type_from_key(key) != "A")
            return -1;

        // Print sub propagator_codes
        std::vector<std::pair<std::string, ComputationEdge>> computation_propagator_codes = propagator_analyzer.get_computation_propagator_codes();
        for(const auto& item : computation_propagator_codes)
        {
            std::cout << item.first << ":\n\t";
            std::cout << "{max_n_segment: " << item.second.max_n_segment << ",\n\tsub_deps: [";
            sub_deps = PropagatorCode::get_deps_from_key(item.first);
            for(size_t i=0; i<sub_deps.size(); i++)
                std::cout << std::get<0>(sub_deps[i]) << ":" << std::get<1>(sub_deps[i]) << ", " ;
            std::cout << "]}" << std::endl;
        }

        // Check size of sub_deps dictionary
        if(propagator_analyzer.get_n_computation_propagator_codes() != 29)
            return -1;

        // Check max_n_segment
        if(propagator_analyzer.get_computation_propagator("B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(B12)B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(B12)A").max_n_segment != 9)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(A12B12)A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(A12)B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((A12B12)A9)A").max_n_segment != 9)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((A12)B12(B12)A9(B12)B12)A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((A12B12)A9)A9(A12B12)A12)A").max_n_segment != 9)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((A12)B12(B12)A9(B12)B12)A12B12B9)A").max_n_segment != 4)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((A12B12)A9)A9(A12B12)A12)A9A12)A").max_n_segment != 4)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A").max_n_segment != 9)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((A12)B12(B12)A9(B12)B12)A12B12B9)A4(((A12B12)A9)A9(A12B12)A12)A9)A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4((A12)B12(B12)A9(B12)B12)A12B9)B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4((A12)B12(B12)A9(B12)B12)A12B12)B").max_n_segment != 9)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A").max_n_segment != 9)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(B12)A9(B12)B12)B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A").max_n_segment != 9)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)A9)B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A").max_n_segment != 9)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A12B12)A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A12A12)B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(B12)A9(B12)B12)B12)A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A9)B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)A9)B12)B").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A9B12)A").max_n_segment != 12)
            return -1;
        if(propagator_analyzer.get_computation_propagator("(((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A9A12)B").max_n_segment != 12)
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
