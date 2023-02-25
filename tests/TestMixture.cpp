#include <iostream>
#include <cmath>

#include "Mixture.h"
#include "PolymerChain.h"

int main()
{
    try
    {
        Mixture mx("Continuous", 0.1, {{"A",1.0}, {"B",2.0}}, false);
        mx.add_polymer(1.0, 
            {"A","A","B","B","A","A","B","A","B","B","A","A","B","A","B","A","A","B","A"},
            {0.4,1.2,1.2,0.9,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,1.2,1.2},
            {0,0,0,0,1,1,2,2,2,3,4,4,7,8,9,9,10,13,13},
            {1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18}, {});

        // display all blocks and branches
        mx.display_unique_blocks();
        mx.display_unique_branches();

        // test get_deps_from_key
        std::string key;
        std::vector<std::tuple<std::string, int, int>> sub_deps;
        // sub_deps: []
        key = "B";
        sub_deps = Mixture::get_deps_from_key(key);
        if(sub_deps.size() != 0)
            return -1;
        if(Mixture::get_monomer_type_from_key(key) != "B")
            return -1;

        // sub_deps: [A:12, B:12]
        key = "(A12B12)A";
        sub_deps = Mixture::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "A" || std::get<1>(sub_deps[0]) != 12)
            return -1;
        if(std::get<0>(sub_deps[1]) != "B" || std::get<1>(sub_deps[1]) != 12)
            return -1;
        if(Mixture::get_monomer_type_from_key(key) != "A")
            return -1;

        // sub_deps: [(A12)B:12, (B12)A:9, (B12)B:12]
        key = "((A12)B12(B12)A9(B12)B12)A";
        sub_deps = Mixture::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "(A12)B" || std::get<1>(sub_deps[0]) != 12)
            return -1;
        if(std::get<0>(sub_deps[1]) != "(B12)A" || std::get<1>(sub_deps[1]) != 9)
            return -1;
       if(std::get<0>(sub_deps[2]) != "(B12)B" || std::get<1>(sub_deps[2]) != 12)
            return -1;
        if(Mixture::get_monomer_type_from_key(key) != "A")
            return -1;

        // sub_deps: [(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A:12, (A12)B:12, (B12)B:12]}
        key = "((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A";
        sub_deps = Mixture::get_deps_from_key(key);
        if(std::get<0>(sub_deps[0]) != "(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A" || std::get<1>(sub_deps[0]) != 12)
            return -1;
        if(std::get<0>(sub_deps[1]) != "(A12)B" || std::get<1>(sub_deps[1]) != 12)
            return -1;
        if(std::get<0>(sub_deps[2]) != "(B12)B" || std::get<1>(sub_deps[2]) != 12)
            return -1;
        if(Mixture::get_monomer_type_from_key(key) != "A")
            return -1;

        // print unique sub branches
        std::map<std::string, UniqueEdge, CompareBranchKey> unique_branches = mx.get_unique_branches();
        for(const auto& item : unique_branches)
        {
            std::cout << item.first << ":\n\t";
            std::cout << "{max_n_segment: " << item.second.max_n_segment << ",\n\tsub_deps: [";
            sub_deps = Mixture::get_deps_from_key(item.first);
            for(size_t i=0; i<sub_deps.size(); i++)
                std::cout << std::get<0>(sub_deps[i]) << ":" << std::get<1>(sub_deps[i]) << ", " ;
            std::cout << "]}" << std::endl;
        }

        // check size of sub_deps dictionary
        if(mx.get_unique_n_branches() != 29)
            return -1;

        // check max_n_segment
        if(mx.get_unique_branch("B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(B12)B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(B12)A").max_n_segment != 9)
            return -1;
        if(mx.get_unique_branch("(A12B12)A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(A12)B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("((A12B12)A9)A").max_n_segment != 9)
            return -1;
        if(mx.get_unique_branch("((A12)B12(B12)A9(B12)B12)A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((A12B12)A9)A9(A12B12)A12)A").max_n_segment != 9)
            return -1;
        if(mx.get_unique_branch("(((A12)B12(B12)A9(B12)B12)A12B12B9)A").max_n_segment != 4)
            return -1;
        if(mx.get_unique_branch("((((A12B12)A9)A9(A12B12)A12)A9A12)A").max_n_segment != 4)
            return -1;
        if(mx.get_unique_branch("((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A").max_n_segment != 9)
            return -1;
        if(mx.get_unique_branch("((((A12)B12(B12)A9(B12)B12)A12B12B9)A4(((A12B12)A9)A9(A12B12)A12)A9)A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4((A12)B12(B12)A9(B12)B12)A12B9)B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4((A12)B12(B12)A9(B12)B12)A12B12)B").max_n_segment != 9)
            return -1;
        if(mx.get_unique_branch("(((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A").max_n_segment != 9)
            return -1;
        if(mx.get_unique_branch("(((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(B12)A9(B12)B12)B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A").max_n_segment != 9)
            return -1;
        if(mx.get_unique_branch("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)A9)B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A").max_n_segment != 9)
            return -1;
        if(mx.get_unique_branch("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A12B12)A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A12A12)B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(B12)A9(B12)B12)B12)A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A9)B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)A9)B12)B").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A9B12)A").max_n_segment != 12)
            return -1;
        if(mx.get_unique_branch("(((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A9A12)B").max_n_segment != 12)
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
