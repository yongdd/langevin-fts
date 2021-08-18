#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "CircularBuffer.h"

int main()
{

    const int MM{5};
    const int SIZE{3};
    std::stringstream ss;
    std::vector<std::string> vec_string;

    // arrays to calculate anderson mixing
    CircularBuffer cb(SIZE, MM);
    double * p_arr;

    p_arr = new double[MM] {1,2,3,5,4};
    cb.insert(p_arr);
    delete[] p_arr;
    ss << "cb.get_array";
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        ss << "," ;
        for(int j=0; j<MM; j++)
            ss << p_arr[j];
    }
    vec_string.push_back(ss.str());
    ss.str("");

    p_arr = new double[MM] {4,2,1,1,2};
    cb.insert( p_arr );
    delete[] p_arr;
    ss << "cb.get_array";
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        ss << "," ;
        for(int j=0; j<MM; j++)
            ss << p_arr[j];
    }
    vec_string.push_back(ss.str());
    ss.str("");

    p_arr = new double[MM] {3,2,1,5,4};
    cb.insert( p_arr );
    delete[] p_arr;
    ss << "cb.get_array";
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        ss << "," ;
        for(int j=0; j<MM; j++)
            ss << p_arr[j];
    }
    vec_string.push_back(ss.str());
    ss.str("");

    p_arr = new double[MM] {5,4,3,1,2};
    cb.insert( p_arr );
    delete[] p_arr;
    ss << "cb.get_array";
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        ss << "," ;
        for(int j=0; j<MM; j++)
            ss << p_arr[j];
    }
    vec_string.push_back(ss.str());
    ss.str("");

    p_arr = new double[MM] {2,5,1,4,3};
    cb.insert( p_arr );
    delete[] p_arr;
    ss << "cb.get_array";
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        ss << "," ;
        for(int j=0; j<MM; j++)
            ss << p_arr[j];
    }
    vec_string.push_back(ss.str());
    ss.str("");

    ss << "cb.get";
    for(int i=0; i<SIZE; i++)
    {
        ss << "," ;
        for(int j=0; j<MM; j++)
            ss << cb.get(i,j);
    }
    vec_string.push_back(ss.str());
    ss.str("");

    ss << "cb.get_sym";
    for(int i=0; i<SIZE; i++)
    {
        ss << "," ;
        for(int j=0; j<MM; j++)
            ss << cb.get_sym(i,j);
    }
    vec_string.push_back(ss.str());
    ss.str("");

    for(unsigned int i=0; i<vec_string.size(); i++)
        std::cout<< vec_string[i] << std::endl;

    if(vec_string[0] != "cb.get_array,12354,00000,00000")
        return -1;
    if(vec_string[1] != "cb.get_array,12354,42112,00000")
        return -1;
    if(vec_string[2] != "cb.get_array,12354,42112,32154")
        return -1;
    if(vec_string[3] != "cb.get_array,42112,32154,54312")
        return -1;
    if(vec_string[4] != "cb.get_array,32154,54312,25143")
        return -1;
    if(vec_string[5] != "cb.get,32154,54312,25143")
        return -1;
    if(vec_string[6] != "cb.get_sym,34152,45511,15223")
        return -1;
    return 0;
}
