#include <iostream>
#include <cmath>
#include "BranchedPolymerChain.h"

int main()
{
    try
    {
        // BranchedPolymerChain pc("Continuous", 0.1, {{"A",1.0}, {"B",2.0}}, {"A","B","A"}, {0.3, 1.4, 0.4}, {0, 1, 2}, {1, 2, 3});
        BranchedPolymerChain pc("Continuous", 0.1, {{"A",1.0}, {"B",2.0}},
        {"A","A","B","B","A","A","B","A","B","B","A","A","B","A","B","A","A","B","A"},
        {0.4,1.2,1.2,0.9,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,1.2,1.2},
        {0,0,0,0,1,1,2,2,2,3,4,4,7,8,9,9,10,13,13},
        {1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18},
        {});

        // test key_to_deps
        std::string key;
        std::vector<std::pair<std::string, int>> sub_deps;
        // sub_deps: []
        key = "B";
        sub_deps = pc.key_to_deps(key);
        if(sub_deps.size() != 0)
            return -1;
        if(pc.key_to_species(key) != "B")
            return -1;

        // sub_deps: [A:12, B:12]
        key = "(A12B12)A";
        sub_deps = pc.key_to_deps(key);
        if(sub_deps[0].first != "A" || sub_deps[0].second != 12)
            return -1;
        if(sub_deps[1].first != "B" || sub_deps[1].second != 12)
            return -1;
        if(pc.key_to_species(key) != "A")
            return -1;

        // sub_deps: [(A12)B:12, (B12)A:9, (B12)B:12]
        key = "((A12)B12(B12)A9(B12)B12)A";
        sub_deps = pc.key_to_deps(key);
        if(sub_deps[0].first != "(A12)B" || sub_deps[0].second != 12)
            return -1;
        if(sub_deps[1].first != "(B12)A" || sub_deps[1].second != 9)
            return -1;
       if(sub_deps[2].first != "(B12)B" || sub_deps[2].second != 12)
            return -1;
        if(pc.key_to_species(key) != "A")
            return -1;

        // sub_deps: [(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A:12, (A12)B:12, (B12)B:12]}
        key = "((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A";
        sub_deps = pc.key_to_deps(key);
        if(sub_deps[0].first != "(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A" || sub_deps[0].second != 12)
            return -1;
        if(sub_deps[1].first != "(A12)B" || sub_deps[1].second != 12)
            return -1;
       if(sub_deps[2].first != "(B12)B" || sub_deps[2].second != 12)
            return -1;
        if(pc.key_to_species(key) != "A")
            return -1;

        // print unique sub branches
        std::map<std::string, int, std::greater<std::string>> opt_max_segments = pc.get_opt_max_segments();
        for(const auto& item : opt_max_segments){
            std::cout << item.first << ":\n\t";
            std::cout << "{max_segments: " << item.second << ",\n\tsub_deps: [";
            sub_deps = pc.key_to_deps(item.first);
            for(int i=0; i<sub_deps.size(); i++)
                std::cout << sub_deps[i].first << ":" << sub_deps[i].second << ", " ;
            std::cout << "]}" << std::endl;
        }

        // check size of sub_deps dictionary
        if(pc.get_opt_n_branches() != 29)
            return -1;

        // check max_segments
        if(pc.get_opt_max_segment("B") != 12)
            return -1;
        if(pc.get_opt_max_segment("A") != 12)
            return -1;
        if(pc.get_opt_max_segment("(B12)B") != 12)
            return -1;
        if(pc.get_opt_max_segment("(B12)A") != 9)
            return -1;
        if(pc.get_opt_max_segment("(A12B12)A") != 12)
            return -1;
        if(pc.get_opt_max_segment("(A12)B") != 12)
            return -1;
        if(pc.get_opt_max_segment("((A12B12)A9)A") != 9)
            return -1;
        if(pc.get_opt_max_segment("((A12)B12(B12)A9(B12)B12)A") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((A12B12)A9)A9(A12B12)A12)A") != 9)
            return -1;
        if(pc.get_opt_max_segment("(((A12)B12(B12)A9(B12)B12)A12B12B9)A") != 4)
            return -1;
        if(pc.get_opt_max_segment("((((A12B12)A9)A9(A12B12)A12)A9A12)A") != 4)
            return -1;
        if(pc.get_opt_max_segment("((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A") != 9)
            return -1;
        if(pc.get_opt_max_segment("((((A12)B12(B12)A9(B12)B12)A12B12B9)A4(((A12B12)A9)A9(A12B12)A12)A9)A") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4((A12)B12(B12)A9(B12)B12)A12B9)B") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4((A12)B12(B12)A9(B12)B12)A12B12)B") != 9)
            return -1;
        if(pc.get_opt_max_segment("(((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A") != 9)
            return -1;
        if(pc.get_opt_max_segment("(((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A") != 12)
            return -1;
        if(pc.get_opt_max_segment("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(B12)A9(B12)B12)B") != 12)
            return -1;
        if(pc.get_opt_max_segment("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A") != 9)
            return -1;
        if(pc.get_opt_max_segment("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)A9)B") != 12)
            return -1;
        if(pc.get_opt_max_segment("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A") != 9)
            return -1;
        if(pc.get_opt_max_segment("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A12B12)A") != 12)
            return -1;
        if(pc.get_opt_max_segment("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A12A12)B") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(B12)A9(B12)B12)B12)A") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A9)B") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)A9)B12)B") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A9B12)A") != 12)
            return -1;
        if(pc.get_opt_max_segment("(((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A9A12)B") != 12)
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
