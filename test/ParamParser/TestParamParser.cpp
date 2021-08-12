#include "ParamParser.h"

int main()
{
    int i;
    double d;
    std::string s;
    ParamParser pp("inputs");
    pp.get("geometry.grids", i);
    std::cout<< "geometry.grids[0]: "<< i << std::endl;

    pp.get("chain.chiN", d);
    std::cout<< "chain.chiN: "<< d << std::endl;

    pp.get("filename", s);
    std::cout<< "filename: "<< s << std::endl;

    return 0;
}
