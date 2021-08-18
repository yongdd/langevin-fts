
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include "RandomGaussian.h"

int main()
{
    const double PI = 3.14159265358979323846;
    const int MM = 10000000;
    const int histMM = 200;
    const double infty = std::numeric_limits<double>::infinity();
    double normal_noise[MM] {0.0};
    int histo_count[2*histMM+1];
    double histo[2*histMM+1];
    double dist[2*histMM+1];
    unsigned int seed = 0;
    double mean, sigma, noise;
    double mean_out, sigma_out;
    int i, k;
    double x, bin_width, error, total;

    // initialize
    mean = 5.0;
    sigma = 13.0;
    bin_width = 0.2;
    RandomGaussian rg;

    // run
    for(int i=0; i<MM; i++)
    {
        do
        {
            noise = rg.normal_dist(mean, sigma);
            if(noise < infty)
                break;
            else
                std::cout<< "Infinity in normal_noise, retry..." << std::endl;
        }
        while(true);
        normal_noise[i] = noise;
    }

    // Mean
    total=0.0;
    for(int i=0; i<MM; i++)
        total += normal_noise[i];
    mean_out = total/MM;

    // Standard deviation
    total=0.0;
    for(int i=0; i<MM; i++)
        total += pow(normal_noise[i]-mean_out,2);
    sigma_out = sqrt(total/MM);

    // Histogram
    for(int i=-histMM; i<=histMM; i++)
        histo_count[i+histMM] = 0;
    for(int i=0; i<MM; i++)
    {
        k = std::lround(normal_noise[i]/bin_width);
        if ( -histMM <= k && k <= histMM )
            histo_count[k + histMM]++;
    }
    for(int i=-histMM; i<=histMM; i++)
        histo[i + histMM] = ((double) histo_count[i + histMM])/(MM*bin_width);

    // Gaussian distribution
    for(int i=-histMM; i<=histMM; i++)
    {
        x= i*bin_width;
        dist[i + histMM] = exp(-pow(x-mean,2)/(2*pow(sigma,2)))/sigma/sqrt(2*PI);
    }

    // Error
    error = 0.0;
    for(int i=-histMM; i<=histMM; i++)
        error += pow(histo[i + histMM] - dist[i + histMM],2);
    error = sqrt(error)*bin_width;

    // write output
    std::cout<< "Test 1: " << MM << std::endl;
    std::cout<< "mean: " << mean <<  ", " << mean_out << std::endl;
    std::cout<< "sigma: " << sigma <<  ", " << sigma_out << std::endl;
    std::cout<< "error: " << error << std::endl;

    // write file
    std::string file_name = "TestRandomGaussian_1.txt";
    std::ofstream write_file(file_name.data());
    if( write_file.is_open() )
    {
        for(int i=-histMM; i<=histMM; i++)
        {
            x= i*bin_width;
            write_file<< i << " " << x << " " << histo[i + histMM]
                        << " " << dist[i + histMM] << std::endl;
        }
        write_file.close();
    }

    //std::cout<< "std::abs(sigma-sigma_out) " <<  std::abs(sigma-sigma_out) << std::endl;

    if( std::abs(mean-mean_out) > 5e-3)
        return -1;
    if( std::abs(sigma-sigma_out) > 1e-3)
        return -1;
    if( error > 1e-3)
        return -1;

    return 0;
}
