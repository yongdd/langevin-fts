#include <iostream>
#include <cmath>

#include "Mixture.h"
#include "PolymerChain.h"

int main()
{
    try
    {
        Mixture mx("discrete", 0.1, {{"A",1.0}, {"B",1.0}}, false);
        mx.add_polymer(1.0, 
            {"A","A","A","A","A","B","B","B","B","B","B","A","A"},
            {0.2,0.2,0.2,0.2,0.2,0.8,0.8,0.4,0.2,0.8,0.8,0.2,0.2},
            {0,1,2,3,4,1,2,3,4,6,7,6,7},
            {1,2,3,4,5,6,7,8,9,10,12,11,13}, {});

        // display all blocks and branches
        mx.display_unique_blocks();
        mx.display_unique_branches();

        // test key_to_deps
        std::string key;
        std::vector<std::tuple<std::string, int, int>> sub_deps;


        std::cout << "-------test---------------------------" << std::endl;
        // sub_deps[
        //     (((((A2B8)B8A2)A2(A2B8)B8)A2B4)A2B2)A:5:2,
        //     ((((A2B2)A2B4)A2(A2B8)B8)A2(A2B8)B8)A:7:3,
        //     [[(((A2B2)A2B4)A2((A2B8)B8A2)A2)B1:1,((((A2B2)A2B4)A2(A2B8)B8)A2A2)B1:1]B7B2]A:5:1,
        key = "[[[(((A2B2)A2B4)A2((A2B8)B8A2)A2)B1:1,((((A2B2)A2B4)A2(A2B8)B8)A2A2)B1:1]B7B2]A5,((((A2B2)A2B4)A2(A2B8)B8)A2(A2B8)B8)A7:3,(((((A2B8)B8A2)A2(A2B8)B8)A2B4)A2B2)A5:2]A";
        sub_deps = Mixture::key_to_deps(key);
        if(std::get<0>(sub_deps[0]) != "[[(((A2B2)A2B4)A2((A2B8)B8A2)A2)B1:1,((((A2B2)A2B4)A2(A2B8)B8)A2A2)B1:1]B7B2]A"
            || std::get<1>(sub_deps[0]) != 5 || std::get<2>(sub_deps[0]) != 1)
            return -1;
        if(std::get<0>(sub_deps[1]) != "((((A2B2)A2B4)A2(A2B8)B8)A2(A2B8)B8)A"
            || std::get<1>(sub_deps[1]) != 7 || std::get<2>(sub_deps[1]) != 3)
            return -1;
        if(std::get<0>(sub_deps[2]) != "(((((A2B8)B8A2)A2(A2B8)B8)A2B4)A2B2)A"
            || std::get<1>(sub_deps[2]) != 5 || std::get<2>(sub_deps[2]) != 2)
            return -1;
        if(Mixture::key_to_species(key) != "A")
            return -1;

        for (const auto& item: sub_deps)
            std::cout << std::get<0>(item) + ", " + std::to_string(std::get<1>(item)) + ", " + std::to_string(std::get<2>(item)) << std::endl;

        key = "[[(((A2B2)A2B4)A2((A2B8)B8A2)A2)B1:1,((((A2B2)A2B4)A2(A2B8)B8)A2A2)B1:1]B7A8]B";
        sub_deps = Mixture::key_to_deps(key);
        if(std::get<0>(sub_deps[0]) != "[(((A2B2)A2B4)A2((A2B8)B8A2)A2)B1:1,((((A2B2)A2B4)A2(A2B8)B8)A2A2)B1:1]B"
            || std::get<1>(sub_deps[0]) != 7 || std::get<2>(sub_deps[0]) != 1)
            return -1;
        if(std::get<0>(sub_deps[1]) != "A"
            || std::get<1>(sub_deps[1]) != 8 || std::get<2>(sub_deps[1]) != 1)
            return -1;
        if(Mixture::key_to_species(key) != "B")
            return -1;

        for (const auto& item: sub_deps)
            std::cout << std::get<0>(item) + ", " + std::to_string(std::get<1>(item)) + ", " + std::to_string(std::get<2>(item)) << std::endl;

        key = "([(((A2B2)A2B4)A2((A2B8)B8A2)A2)B1:1,((((A2B2)A2B4)A2(A2B8)B8)A2A2)B1:1]B7A8)B";
        sub_deps = Mixture::key_to_deps(key);
        // if(std::get<0>(sub_deps[0]) != "[[(((A2B2)A2B4)A2((A2B8)B8A2)A2)B1:1,((((A2B2)A2B4)A2(A2B8)B8)A2A2)B1:1]B7B2]A"
        //     || std::get<1>(sub_deps[0]) != 5 || std::get<2>(sub_deps[0]) != 1)
        //     return -1;
        // if(std::get<0>(sub_deps[1]) != "((((A2B2)A2B4)A2(A2B8)B8)A2(A2B8)B8)A"
        //     || std::get<1>(sub_deps[1]) != 7 || std::get<2>(sub_deps[1]) != 3)
        //     return -1;
        // if(std::get<0>(sub_deps[2]) != "(((((A2B8)B8A2)A2(A2B8)B8)A2B4)A2B2)A"
        //     || std::get<1>(sub_deps[2]) != 5 || std::get<2>(sub_deps[2]) != 2)
        //     return -1;
        // if(Mixture::key_to_species(key) != "A")
        for (const auto& item: sub_deps)
            std::cout << std::get<0>(item) + ", " + std::to_string(std::get<1>(item)) + ", " + std::to_string(std::get<2>(item)) << std::endl;



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
