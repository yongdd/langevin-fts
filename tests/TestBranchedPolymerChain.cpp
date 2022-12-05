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
        {1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18});

        // // print unique sub branches
        // std::map<std::string, polymer_chain_optimal_sub_branches_data, std::greater<std::string>> optimal_sub_branches = pc.get_optimal_sub_branches();
        // for(const auto& item : optimal_sub_branches){
        //     std::cout << item.first << ":\n\t";
        //     std::cout << "{max_segments: " << item.second.max_segment << ",\n\tdependencies: [";
        //     int count = item.second.dependency.size(); 
        //     if(count > 0){
        //         for(int i=0; i<count-1; i++){ 
        //             std::cout << "[" << item.second.dependency[i].first << ", " <<  item.second.dependency[i].second << "], " ;
        //         }
        //         std::cout << "[" << item.second.dependency[count-1].first << ", " <<  item.second.dependency[count-1].second << "]" ;
        //     }
        //     std::cout << "]}" << std::endl;
        // }

        // check size of dependency dictionary
        if(pc.get_n_sub_graph() != 29)
            return -1;

        // check max_segments
        if(pc.get_max_segment("B") != 12)
            return -1;
        if(pc.get_max_segment("A") != 12)
            return -1;
        if(pc.get_max_segment("(B12)B") != 12)
            return -1;
        if(pc.get_max_segment("(B12)A") != 9)
            return -1;
        if(pc.get_max_segment("(A12B12)A") != 12)
            return -1;
        if(pc.get_max_segment("(A12)B") != 12)
            return -1;
        if(pc.get_max_segment("((A12B12)A9)A") != 9)
            return -1;
        if(pc.get_max_segment("((A12)B12(B12)A9(B12)B12)A") != 12)
            return -1;
        if(pc.get_max_segment("(((A12B12)A9)A9(A12B12)A12)A") != 9)
            return -1;
        if(pc.get_max_segment("(((A12)B12(B12)A9(B12)B12)A12B12B9)A") != 4)
            return -1;
        if(pc.get_max_segment("((((A12B12)A9)A9(A12B12)A12)A9A12)A") != 4)
            return -1;
        if(pc.get_max_segment("((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A") != 9)
            return -1;
        if(pc.get_max_segment("((((A12)B12(B12)A9(B12)B12)A12B12B9)A4(((A12B12)A9)A9(A12B12)A12)A9)A") != 12)
            return -1;
        if(pc.get_max_segment("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A") != 12)
            return -1;
        if(pc.get_max_segment("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4((A12)B12(B12)A9(B12)B12)A12B9)B") != 12)
            return -1;
        if(pc.get_max_segment("(((((A12B12)A9)A9(A12B12)A12)A9A12)A4((A12)B12(B12)A9(B12)B12)A12B12)B") != 9)
            return -1;
        if(pc.get_max_segment("(((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A") != 9)
            return -1;
        if(pc.get_max_segment("(((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A") != 12)
            return -1;
        if(pc.get_max_segment("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(B12)A9(B12)B12)B") != 12)
            return -1;
        if(pc.get_max_segment("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A") != 9)
            return -1;
        if(pc.get_max_segment("((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)A9)B") != 12)
            return -1;
        if(pc.get_max_segment("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A") != 9)
            return -1;
        if(pc.get_max_segment("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A12B12)A") != 12)
            return -1;
        if(pc.get_max_segment("((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9((A12B12)A9)A9)A12A12)B") != 12)
            return -1;
        if(pc.get_max_segment("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(B12)A9(B12)B12)B12)A") != 12)
            return -1;
        if(pc.get_max_segment("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)B12)A9)B") != 12)
            return -1;
        if(pc.get_max_segment("(((((((A12B12)A9)A9(A12B12)A12)A9A12)A4B12B9)A12(A12)B12(B12)A9)B12)B") != 12)
            return -1;
        if(pc.get_max_segment("(((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A9B12)A") != 12)
            return -1;
        if(pc.get_max_segment("(((((((A12)B12(B12)A9(B12)B12)A12B12B9)A4A12)A9(A12B12)A12)A9)A9A12)B") != 12)
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
