#include <iostream>
#include <cmath>

#include "Mixture.h"
#include "PolymerChain.h"

int main()
{
    try
    {
        Mixture mx("Continuous", 0.1, {{"A",1.0}, {"B",1.0}});
        mx.add_polymer(1.0, 
            {"A","A","A","A","A","B","B","B","B","B","B","A","A"},
            {0.2,0.2,0.2,0.2,0.2,0.4,0.8,0.4,0.2,0.8,0.8,0.2,0.2},
            {0,1,2,3,4,1,2,3,4,6,7,6,7},
            {1,2,3,4,5,6,7,8,9,10,12,11,13}, {});

        // display all blocks and branches
        mx.display_unique_blocks();
        mx.display_unique_branches();

        // test key_to_deps
        std::string key;
        std::vector<std::tuple<std::string, int, int>> sub_deps;
        // sub_deps: [((((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B4+)B4,(((A2B8)B4A2)A2(A2B8)B8)A2(A2B2)A2:1+)B:2:1,
        //            ((((A2B8)B4A2)A2(A2B8)B8)A2B4)A:2:1]
        // key = "(((((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B4+)B4,(((A2B8)B4A2)A2(A2B8)B8)A2(A2B2)A2:1+)B2,((((A2B8)B4A2)A2(A2B8)B8)A2B4)A2A2:1+)B";
        // sub_deps = Mixture::key_to_deps(key);
        // if(std::get<0>(sub_deps[0]) != "((((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B4+)B4,(((A2B8)B4A2)A2(A2B8)B8)A2(A2B2)A2:1+)B"
        //     || std::get<1>(sub_deps[0]) != 2 || std::get<2>(sub_deps[0]) != 1)
        //     return -1;
        // if(std::get<0>(sub_deps[1]) != "((((A2B8)B4A2)A2(A2B8)B8)A2B4)A" || std::get<1>(sub_deps[1]) != 2 || std::get<2>(sub_deps[1]) != 1)
        //     return -1;
        // if(Mixture::key_to_species(key) != "B")
        //     return -1;

        // for (const auto& item: sub_deps)
        //     std::cout << std::get<0>(item) + ", " + std::to_string(std::get<1>(item)) + ", " + std::to_string(std::get<2>(item)) << std::endl;

        // // sub_deps: [((((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B4+)B:4:1,
        // //            (((A2B8)B4A2)A2(A2B8)B8)A2(A2B2)A:2:1]
        // key = "((((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B4+)B4,(((A2B8)B4A2)A2(A2B8)B8)A2(A2B2)A2:1+)B";
        // sub_deps = Mixture::key_to_deps(key);
        // if(std::get<0>(sub_deps[0]) != "((((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B4+)B"
        //     || std::get<1>(sub_deps[0]) != 4 || std::get<2>(sub_deps[0]) != 1)
        //     return -1;
        // if(std::get<0>(sub_deps[1]) != "(((A2B8)B4A2)A2(A2B8)B8)A2(A2B2)A" || std::get<1>(sub_deps[1]) != 2 || std::get<2>(sub_deps[1]) != 1)
        //     return -1;
        // if(Mixture::key_to_species(key) != "B")
        //     return -1;

        // // sub_deps: [((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B:4:1]
        // key = "((((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B4+)B";
        // sub_deps = Mixture::key_to_deps(key);
        // if(std::get<0>(sub_deps[0]) != "((((A2B2)A2B4)A2((A2B8)B4A2)A2)B4:1,(((A2B2)A2B4)A2(A2B8)B8)A2A2:1+)B"
        //     || std::get<1>(sub_deps[0]) != 4 || std::get<2>(sub_deps[0]) != 1)
        //     return -1;
        // if(Mixture::key_to_species(key) != "B")
        //     return -1;

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
