#include <iostream>
#include <string>
#include <map>

#include "CompareBranchKey.h"

int main()
{
    try
    {
        std::string str1;
        std::map<std::string, int, CompareBranchKey> map;

        map["A"] = 0;
        map["B"] = 1;
        map["AA"] = 2;
        map["(A)"] = 3;
        map["[A]"] = 4;
        map["(B)"] = 5;
        map["[B]"] = 6;
        map["a"] = 7;
        map["b"] = 8;

        std::vector<std::string> vect;
        for(const auto& item: map)
            vect.push_back(item.first);

        for(const auto& item: vect)
            std::cout << item << std::endl;
        
        if(vect[0] != "A")
            return -1;
        if(vect[1] != "AA")
            return -1;
        if(vect[2] != "B")
            return -1;
        if(vect[3] != "a")
            return -1;
        if(vect[4] != "b")
            return -1;
        if(vect[5] != "(A)")
            return -1;
        if(vect[6] != "(B)")
            return -1;
        if(vect[7] != "[A]")
            return -1;
        if(vect[8] != "[B]")
            return -1;

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
