
#include <string>
#include <array>
#include <vector>
#include "Exception.h"
#include "ParamParser.h"

int main()
{
    try
    {
        int nx0;
        std::vector<int> nx = {1,2,3};
        double chi_n;
        std::string filename;
        std::vector<std::string> as_string;
        
        ParamParser &pp = ParamParser::get_instance();
        pp.read_param_file("TestInputParams",false);
        pp.get("geometry.grids", nx0);
        pp.get("geometry.grids", nx);
        pp.get("chain.chi_n", chi_n);
        pp.get("output.filename", filename);
        as_string = pp.get("chain.chi_n");
        
        //std::cout<< "geometry.grids[0]: "<< nx0 << std::endl;
        std::cout<< "geometry.grids: ";
        std::cout<< nx[0] << " "<< nx[1] << " "<< nx[2] << std::endl;
        std::cout<< "chain.chi_n: "<< chi_n << std::endl;
        std::cout<< "output.filename: "<< filename << std::endl;
        std::cout<< "as_string: "<< as_string[0] << std::endl;
        
        if( nx0 != 31)
            return -1;
        if( nx[0] != 31 || nx[1] != 49 || nx[2] != 63)
            return -1;
        if( (chi_n - 20.0) > 1e-7)
            return -1;
        if( filename != "fields")
            return -1;
        if( as_string[0] != "20.0")
            return -1;
        
        //pp.display_usage_info();

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
