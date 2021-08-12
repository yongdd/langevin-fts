#include "ParamParser.h"

int main()
{
    int i, nx[3];
    double d;
    std::string s;
    ParamParser pp("inputs");
    pp.get("geometry.grids", i);
    std::cout<< "geometry.grids[0]: "<< i << std::endl;

    pp.get("geometry.grids", nx, 3);
    std::cout<< "geometry.grids: ";
    std::cout<< nx[0] << " "<< nx[1] << " "<< nx[2] << std::endl;

    pp.get("chain.chiN", d);
    std::cout<< "chain.chiN: "<< d << std::endl;

    pp.get("filename", s);
    std::cout<< "filename: "<< s << std::endl;

    return 0;
}
