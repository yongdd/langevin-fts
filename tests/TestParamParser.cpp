
#include <string>
#include <vector>
#include "ParamParser.h"

int main()
{
    int nx0, nx[3];
    double chi_n;
    std::string filename;
    std::vector<std::string> as_string;
    
    ParamParser &pp = ParamParser::get_instance();
    pp.read_param_file("TestInputParams",false);
    pp.get("geometry.grids", nx0);
    pp.get("geometry.grids", nx, 3);
    pp.get("chain.chi_n", chi_n);
    pp.get("filename", filename);
    as_string = pp.get("chain.chi_n");

    if( nx0 != 24)
        return -1;
    if( nx[0] != 24 || nx[1] != 36 || nx[2] != 48)
        return -1;
    if( (chi_n - 20.0) > 1e-7)
        return -1;
    if( filename != "asdf")
        return -1;
    if( as_string[0] != "20.0")
        return -1;
    
    //std::cout<< "geometry.grids[0]: "<< nx0 << std::endl;
    //std::cout<< "geometry.grids: ";
    //std::cout<< nx[0] << " "<< nx[1] << " "<< nx[2] << std::endl;
    //std::cout<< "chain.chi_n: "<< chi_n << std::endl;
    //std::cout<< "filename: "<< filename << std::endl;
    //std::cout<< "as_string: "<< as_string[0] << std::endl;
    //pp.display_usage_info();

    return 0;
}
