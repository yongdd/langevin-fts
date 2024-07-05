#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

#include "Exception.h"
#include "Array.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

int main()
{
    try
    {
        const int N_REPEATS = 1;

        // Math constants
        const double PI = 3.14159265358979323846;

        // Chrono timer
        std::chrono::system_clock::time_point chrono_start, chrono_end;
        std::chrono::duration<double> time_duration;

        // -------------- initialize ------------
        std::vector<int> nx = {31,49,23};
        std::vector<double> lx = {4.0,3.0,2.0};

        bool reduce_memory_usage = false;
        std::string chain_model = "continuous";

        const int M = nx[0]*nx[1]*nx[2];

        double _array_1[M];
        double _array_2[M];
        double _array_3[M] = {0.0, };

        double _array_1_two[2*M];
        double _array_2_two[2*M];

        // set arbitrary arrays
        for(int i=0; i<M; i++)
        {
            _array_1[i] = sqrt(PI*i);
            _array_2[i] = sqrt(PI*(i+10000));
        }
        for(int i=0; i<M; i++)
        {
            _array_1_two[i] = sqrt(PI*i);
            _array_2_two[i] = sqrt(PI*(i+10000));

            _array_1_two[i+M] = sqrt(PI*(i+20000));
            _array_2_two[i+M] = sqrt(PI*(i+30000));
        }

        // Choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        for(std::string platform : avail_platforms)
        {
            AbstractFactory *factory = PlatformSelector::create_factory(platform, reduce_memory_usage);
            factory->display_info();

            // Create instances and assign to the variables of base classes for the dynamic binding
            ComputationBox *cb = factory->create_computation_box(nx, lx, {});
            
            std::cout<< "---------- Run ----------" << std::endl;
            std::cout<< "iteration, mass error, total partitions, total energy, error level" << std::endl;
            chrono_start = std::chrono::system_clock::now();

            for (int i=0; i<N_REPEATS; i++)
            {
                Array* array_1 = factory->create_array(M);
                Array* array_2 = factory->create_array(M);
                Array* array_3 = factory->create_array(M);

                std::cout << "Test passed, factory->create_array(M)." << std::endl;

                array_1->set_data(_array_1, M);
                array_2->set_data(_array_2, M);
                array_3->set_data(_array_3, M);

                std::cout << "Test passed, array->set_data(_array, M)." << std::endl;

                // Add(const Array& src_1, const Array& src_2)
                array_3->add(*array_1, *array_2);
                for(int i=0; i<M; i++)
                    _array_3[i] = _array_1[i] + _array_2[i];
                for(int i=0; i<M; i++)
                {
                    if (std::abs(_array_3[i] - (*array_3)[i]) > 1e-6)
                    {
                        std::cout << "Test failed, add(const Array& src_1, const Array& src_2)." << std::endl;
                        return false;
                    }
                }
                std::cout << "Test passed, add(const Array& src_1, const Array& src_2)." << std::endl;

                // subtract(const Array& src_1, const Array& src_2)
                array_3->subtract(*array_1, *array_2);
                for(int i=0; i<M; i++)
                    _array_3[i] = _array_1[i] - _array_2[i];
                for(int i=0; i<M; i++)
                {
                    if (std::abs(_array_3[i] - (*array_3)[i]) > 1e-6)
                    {
                        std::cout << "Test failed, subtract(const Array& src_1, const Array& src_2) const." << std::endl;
                        return false;
                    }
                }
                std::cout << "Test passed, subtract(const Array& src_1, const Array& src_2) const." << std::endl; 

                // Multiply(const Array& src_1, const Array& src_2)
                array_3->multiply(*array_1, *array_2);
                for(int i=0; i<M; i++)
                    _array_3[i] = _array_1[i] * _array_2[i];
                for(int i=0; i<M; i++)
                {
                    if (std::abs(_array_3[i] - (*array_3)[i]) > 1e-6)
                    {
                        std::cout << "Test failed, multiply(const Array& src_1, const Array& src_2) const." << std::endl;
                        return false;
                    }
                }
                std::cout << "Test passed, multiply(const Array& src_1, const Array& src_2) const." << std::endl;

                // divide(const Array& src_1, const Array& src_2)
                array_3->divide(*array_1, *array_2);
                for(int i=0; i<M; i++)
                    _array_3[i] = _array_1[i] / _array_2[i];
                for(int i=0; i<M; i++)
                {
                    if (std::abs(_array_3[i] - (*array_3)[i]) > 1e-6)
                    {
                        std::cout << "Test failed, divide(const Array& src_1, const Array& src_2) const." << std::endl;
                        return false;
                    }
                }
                std::cout << "Test passed, divide(const Array& src_1, const Array& src_2) const." << std::endl;


                // Linear_scaling(const Array& src, const double a,  const double b)
                array_3->linear_scaling(*array_1, 4.126806, 5.2342231);
                for(int i=0; i<M; i++)
                    _array_3[i] = 4.126806*(*array_1)[i] + 5.2342231;
                for(int i=0; i<M; i++)
                {
                    if (std::abs(_array_3[i] - (*array_3)[i]) > 1e-6)
                    {
                        std::cout << "Test failed, linear_scaling(const Array& src, const double a,  const double b)." << std::endl;
                        return false;
                    }
                }
                std::cout << "Test passed, linear_scaling(const Array& src, const double a,  const double b)." << std::endl;

                // Integral(Array& g)
                if (std::abs(cb->integral(_array_1) - cb->integral_device(array_1->get_ptr())) > 1e-6)
                {
                    std::cout << "Test failed, integral(Array& g)." << std::endl;
                    return false;
                }
                else
                {
                    std::cout << "Test passed, integral(Array& g)." << std::endl;
                }

                // Inner_product(Array& g, Array& h)
                if (std::abs(cb->inner_product(_array_1, _array_2) - cb->inner_product_device(array_1->get_ptr(), array_2->get_ptr())) > 1e-6)
                {
                    std::cout << "Test failed, inner_product(Array& g, Array& h)." << std::endl;
                    return false;
                }
                else
                {
                    std::cout << "Test passed, inner_product(Array& g, Array& h)." << std::endl;
                }

                // Inner_product_inverse_weight(Array& g, Array& h, Array& w)
                array_3->add(*array_1, *array_2);
                for(int i=0; i<M; i++)
                    _array_3[i] = _array_1[i] + _array_2[i];
                if (std::abs(cb->inner_product_inverse_weight(_array_1, _array_2, _array_3) - cb->inner_product_inverse_weight_device(array_1->get_ptr(), array_2->get_ptr(), array_3->get_ptr())) > 1e-6)
                {
                    std::cout << "Test failed, inner_product_inverse_weight(Array& g, Array& h, Array& w)." << std::endl;
                    return false;
                }
                else
                {
                    std::cout << "Test passed, inner_product_inverse_weight(Array& g, Array& h, Array& w)." << std::endl;
                }

                // Multi_inner_product(int n_comp,  Array& g, Array& h)
                Array* array_1_two = factory->create_array(_array_1_two, 2*M);
                Array* array_2_two = factory->create_array(_array_2_two, 2*M);
                if (std::abs(cb->multi_inner_product(2, _array_1_two, _array_2_two) - cb->multi_inner_product_device(2, array_1_two->get_ptr(), array_2_two->get_ptr())) > 1e-6)
                {
                    std::cout << "Test failed, multi_inner_product(int n_comp,  Array& g, Array& h)." << std::endl;
                    return false;
                }
                else
                {
                    std::cout << "Test passed, multi_inner_product(int n_comp,  Array& g, Array& h)." << std::endl;
                }

                // zero_mean(Array& g);
                cb->zero_mean(_array_1);
                cb->zero_mean_device(array_1->get_ptr());
                for(int i=0; i<M; i++)
                {
                    if (std::abs(_array_1[i] - (*array_1)[i]) > 1e-6)
                    {
                        std::cout << "Test failed, zero_mean(Array& g)." << std::endl;
                        return false;
                    }
                }
                std::cout << "Test passed, zero_mean(Array& g)." << std::endl;

                delete cb;

                delete array_1;
                delete array_2;
                delete array_3;

                delete array_1_two;
                delete array_2_two;
            }

            // Estimate execution time
            chrono_end = std::chrono::system_clock::now();
            time_duration = chrono_end - chrono_start;
            std::cout<< "total time: ";
            std::cout<< time_duration.count() << std::endl;
        }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}