
#include <iostream>
#include "CircularBuffer.h"

int main()
{

    const int MM{5};
    const int SIZE{3};

    // arrays to calculate anderson mixing
    CircularBuffer cb(SIZE, MM);
    double * p_arr;

    p_arr = new double[MM] {1,2,3,5,4};
    cb.insert( p_arr );
    delete[] p_arr;
    std::cout<< "cb.get_array" << std::endl;
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        std::cout<< i << ": " ;
        for(int j=0; j<MM; j++)
            std::cout<< p_arr[j] << " ";
        std::cout<< std::endl;
    }
    

    p_arr = new double[MM] {4,2,1,1,2};
    cb.insert( p_arr );
    delete[] p_arr;
    std::cout<< "cb.get_array" << std::endl;
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        std::cout<< i << ": " ;
        for(int j=0; j<MM; j++)
            std::cout<< p_arr[j] << " ";
        std::cout<< std::endl;
    }
    
    p_arr = new double[MM] {3,2,1,5,4};
    cb.insert( p_arr );
    delete[] p_arr;
    std::cout<< "cb.get_array" << std::endl;
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        std::cout<< i << ": " ;
        for(int j=0; j<MM; j++)
            std::cout<< p_arr[j] << " ";
        std::cout<< std::endl;
    }


    p_arr = new double[MM] {5,4,3,1,2};
    cb.insert( p_arr );
    delete[] p_arr;
    std::cout<< "cb.get_array" << std::endl;
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        std::cout<< i << ": " ;
        for(int j=0; j<MM; j++)
            std::cout<< p_arr[j] << " ";
        std::cout<< std::endl;
    }


    p_arr = new double[MM] {2,5,1,4,3};
    cb.insert( p_arr );
    delete[] p_arr;
    std::cout<< "cb.get_array" << std::endl;
    for(int i=0; i<SIZE; i++)
    {
        p_arr = cb.get_array(i);
        std::cout<< i << ": " ;
        for(int j=0; j<MM; j++)
            std::cout<< p_arr[j] << " ";
        std::cout<< std::endl;
    }


    std::cout<< "cb.get" << std::endl;
    for(int i=0; i<SIZE; i++)
    {
        std::cout<< i << ": " ;
        for(int j=0; j<MM; j++)
            std::cout<< cb.get(i,j) << " ";
        std::cout<< std::endl;
    }

    std::cout<< "cb.get_sym" << std::endl;
    for(int i=0; i<SIZE; i++)
    {
        std::cout<< i << ": " ;
        for(int j=0; j<MM; j++)
            std::cout<< cb.get_sym(i,j) << " ";
        std::cout<< std::endl;
    }
}
