#include "mixed_boundary_fft_mkl.h"

// Demo and testing
int main() {
    std::cout << "Multi-Dimensional Mixed Boundary FFT with MKL (Optimized)\n";
    std::cout << "=========================================================\n\n";
    
    try {
        // Example 1: 1D case
        {
            std::cout << "Example 1: 1D Transform (8 points)\n";
            std::cout << "-----------------------------------\n";
            
            std::vector<MKL_INT> dims = {8};
            std::vector<BoundaryType> boundaries = {BoundaryType::REFLECTING};
            
            MixedBoundaryFFT_MKL fft(dims, boundaries);
            fft.printConfig();
            
            std::vector<double> signal(8);
            for (int i = 0; i < 8; ++i) {
                signal[i] = cos(2.0 * M_PI * i / 8.0);
            }
            
            std::cout << "\nInput:  ";
            for (int i = 0; i < 8; ++i) {
                std::cout << std::setw(7) << std::fixed 
                         << std::setprecision(3) << signal[i] << " ";
            }
            std::cout << "\n";
            
            std::vector<double> spectrum, reconstructed;
            fft.forward(signal, spectrum);
            
            std::cout << "Output: ";
            for (int i = 0; i < 8; ++i) {
                std::cout << std::setw(7) << std::fixed 
                         << std::setprecision(3) << spectrum[i] << " ";
            }
            std::cout << "\n";
            
            fft.backward(spectrum, reconstructed);
            std::cout << "Recon:  ";
            for (int i = 0; i < 8; ++i) {
                std::cout << std::setw(7) << std::fixed 
                         << std::setprecision(3) << reconstructed[i] << " ";
            }
            
            // Compute error
            double max_err = 0.0;
            for (int i = 0; i < 8; ++i) {
                max_err = std::max(max_err, std::abs(signal[i] - reconstructed[i]));
            }
            std::cout << "\nMax error: " << std::scientific << max_err << "\n\n";
        }
        
        // Example 2: 2D case
        {
            std::cout << "Example 2: 2D Transform (8x8)\n";
            std::cout << "-----------------------------\n";
            
            std::vector<MKL_INT> dims = {8, 8};
            std::vector<BoundaryType> boundaries = {
                BoundaryType::PERIODIC,
                BoundaryType::REFLECTING
            };
            
            MixedBoundaryFFT_MKL fft(dims, boundaries);
            fft.printConfig();
            
            std::vector<double> data(64);
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    data[i * 8 + j] = sin(2.0 * M_PI * i / 8.0) * 
                                     cos(M_PI * j / 8.0);
                }
            }
            
            Timer timer;
            std::vector<double> spectrum, reconstructed;
            fft.forward(data, spectrum);
            fft.backward(spectrum, reconstructed);
            double elapsed = timer.elapsed();
            
            std::cout << "\nTransform time: " << elapsed << " ms\n";
            
            // Check reconstruction error
            double max_error = 0.0;
            for (size_t i = 0; i < data.size(); ++i) {
                max_error = std::max(max_error, 
                                    std::abs(data[i] - reconstructed[i]));
            }
            std::cout << "Max reconstruction error: " << std::scientific 
                     << max_error << std::fixed << "\n\n";
        }
        
        // Example 3: 3D case with all boundary types
        {
            std::cout << "Example 3: 3D Transform (16x16x16)\n";
            std::cout << "-----------------------------------\n";
            
            std::vector<MKL_INT> dims = {16, 16, 16};
            std::vector<BoundaryType> boundaries = {
                BoundaryType::PERIODIC,    // X: periodic
                BoundaryType::REFLECTING,  // Y: reflecting (Neumann)
                BoundaryType::ABSORBING    // Z: absorbing (Dirichlet)
            };
            
            MixedBoundaryFFT_MKL fft(dims, boundaries);
            fft.printConfig();
            
            // Create 3D test data
            std::vector<double> data(16 * 16 * 16);
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    for (int k = 0; k < 16; ++k) {
                        int idx = i * 256 + j * 16 + k;
                        data[idx] = sin(2.0 * M_PI * i / 16.0) * 
                                   cos(M_PI * j / 16.0) *
                                   sin(M_PI * (k + 1) / 17.0);
                    }
                }
            }
            
            std::cout << "\nPerforming 3D transform...\n";
            Timer timer;
            
            std::vector<double> spectrum, reconstructed;
            fft.forward(data, spectrum);
            double fwd_time = timer.elapsed();
            
            timer = Timer();
            fft.backward(spectrum, reconstructed);
            double bwd_time = timer.elapsed();
            
            std::cout << "Forward transform:  " << fwd_time << " ms\n";
            std::cout << "Backward transform: " << bwd_time << " ms\n";
            std::cout << "Total time:         " << (fwd_time + bwd_time) << " ms\n";
            
            // Check reconstruction
            double max_error = 0.0;
            double avg_error = 0.0;
            for (size_t i = 0; i < data.size(); ++i) {
                double err = std::abs(data[i] - reconstructed[i]);
                max_error = std::max(max_error, err);
                avg_error += err;
            }
            avg_error /= data.size();
            
            std::cout << "Max reconstruction error: " << std::scientific << max_error << "\n";
            std::cout << "Avg reconstruction error: " << avg_error << std::fixed << "\n\n";
        }
        
        // Example 4: Larger 3D benchmark
        {
            std::cout << "Example 4: Large 3D Transform (64x64x64)\n";
            std::cout << "-----------------------------------------\n";
            
            std::vector<MKL_INT> dims = {64, 64, 64};
            std::vector<BoundaryType> boundaries = {
                BoundaryType::PERIODIC,
                BoundaryType::PERIODIC,
                BoundaryType::PERIODIC
            };
            
            MixedBoundaryFFT_MKL fft(dims, boundaries);
            fft.printConfig();
            
            std::vector<double> data(64 * 64 * 64);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = sin(2.0 * M_PI * i / data.size());
            }
            
            std::cout << "\nPerforming large 3D transform...\n";
            Timer timer;
            
            std::vector<double> spectrum;
            fft.forward(data, spectrum);
            double time = timer.elapsed();
            
            std::cout << "Transform time: " << time << " ms\n";
            std::cout << "Throughput: " 
                     << (data.size() * sizeof(double) / 1e6) / (time / 1000.0) 
                     << " MB/s\n\n";
        }
        
        std::cout << "All examples completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// Compilation command:
/*
g++ -O3 -std=c++11 mixed_boundary_fft_mkl.cpp -o mixed_fft \
    -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
*/