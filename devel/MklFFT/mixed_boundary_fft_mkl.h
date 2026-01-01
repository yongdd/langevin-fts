#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <numbers>
#include <memory>
#include <stdexcept>
#include <iomanip>
#include <chrono>

#include "mkl.h"
#include "mkl_dfti.h"
#include "mkl_trig_transforms.h"

enum class BoundaryType {
    PERIODIC,   // Standard FFT
    ABSORBING,  // Sine transform (DST)
    REFLECTING  // Cosine transform (DCT)
};

class MixedBoundaryFFT_MKL {
private:
    std::vector<MKL_INT> dims_;
    std::vector<BoundaryType> boundaries_;
    int ndim_;
    MKL_INT total_size_;
    
    // MKL descriptor handles for FFT
    std::vector<DFTI_DESCRIPTOR_HANDLE> fft_handles_;
    
    // Working buffers
    std::vector<double> real_buffer_;
    std::vector<std::complex<double>> complex_buffer_;
    std::vector<double> temp_buffer_;
    bool is_complex_;
    
    // Precomputed sine/cosine tables for DST/DCT
    std::vector<std::vector<double>> sin_tables_;
    std::vector<std::vector<double>> cos_tables_;
    
    // Stride information
    std::vector<MKL_INT> strides_;
    std::vector<MKL_INT> num_transforms_;
    
public:
    MixedBoundaryFFT_MKL(const std::vector<MKL_INT>& dims, 
                         const std::vector<BoundaryType>& boundaries) 
        : dims_(dims), boundaries_(boundaries), ndim_(dims.size()) {
        
        if (dims.size() != boundaries.size()) {
            throw std::invalid_argument("Dimensions and boundaries size mismatch");
        }
        
        total_size_ = 1;
        for (MKL_INT d : dims_) {
            total_size_ *= d;
        }
        
        real_buffer_.resize(total_size_);
        complex_buffer_.resize(total_size_);
        temp_buffer_.resize(total_size_);
        is_complex_ = false;
        
        precomputeStrides();
        precomputeTrigTables();
        setupDescriptors();
    }
    
    ~MixedBoundaryFFT_MKL() {
        cleanup();
    }
    
    // Forward transform (optimized)
    void forward(const std::vector<double>& input, 
                 std::vector<double>& output) {
        if (input.size() != static_cast<size_t>(total_size_)) {
            throw std::invalid_argument("Input size mismatch");
        }
        
        // Copy input to aligned buffer
        for (MKL_INT i = 0; i < total_size_; ++i) {
            real_buffer_[i] = input[i];
        }
        is_complex_ = false;
        
        // Apply transforms dimension by dimension
        for (int dim = 0; dim < ndim_; ++dim) {
            applyForwardTransform(dim);
        }
        
        // Copy result
        output.resize(total_size_);
        if (is_complex_) {
            for (MKL_INT i = 0; i < total_size_; ++i) {
                output[i] = complex_buffer_[i].real();
            }
        } else {
            for (MKL_INT i = 0; i < total_size_; ++i) {
                output[i] = real_buffer_[i];
            }
        }
    }
    
    // Backward transform (optimized)
    void backward(const std::vector<double>& input,
                  std::vector<double>& output) {
        if (input.size() != static_cast<size_t>(total_size_)) {
            throw std::invalid_argument("Input size mismatch");
        }
        
        // Load input - could be from real or complex data
        for (MKL_INT i = 0; i < total_size_; ++i) {
            real_buffer_[i] = input[i];
        }
        is_complex_ = false;
        
        // Apply inverse transforms dimension by dimension
        for (int dim = ndim_ - 1; dim >= 0; --dim) {
            applyBackwardTransform(dim);
        }
        
        // Apply normalization and copy result
        output.resize(total_size_);
        if (is_complex_) {
            for (MKL_INT i = 0; i < total_size_; ++i) {
                output[i] = complex_buffer_[i].real();
            }
        } else {
            for (MKL_INT i = 0; i < total_size_; ++i) {
                output[i] = real_buffer_[i];
            }
        }
    }
    
    void printConfig() const {
        std::cout << "MKL FFT Configuration (Optimized):\n";
        std::cout << "Dimensions: ";
        for (MKL_INT d : dims_) std::cout << d << " ";
        std::cout << "\nBoundary types:\n";
        for (size_t i = 0; i < boundaries_.size(); ++i) {
            std::cout << "  Dim " << i << ": ";
            switch(boundaries_[i]) {
                case BoundaryType::PERIODIC:
                    std::cout << "PERIODIC (FFT)\n";
                    break;
                case BoundaryType::ABSORBING:
                    std::cout << "ABSORBING (DST)\n";
                    break;
                case BoundaryType::REFLECTING:
                    std::cout << "REFLECTING (DCT)\n";
                    break;
            }
        }
        std::cout << "Total size: " << total_size_ << std::endl;
    }
    
private:
    void precomputeStrides() {
        strides_.resize(ndim_);
        num_transforms_.resize(ndim_);
        
        for (int dim = 0; dim < ndim_; ++dim) {
            MKL_INT stride = 1;
            for (int d = dim + 1; d < ndim_; ++d) {
                stride *= dims_[d];
            }
            strides_[dim] = stride;
            num_transforms_[dim] = total_size_ / (dims_[dim] * stride);
        }
    }
    
    void precomputeTrigTables() {
        sin_tables_.resize(ndim_);
        cos_tables_.resize(ndim_);
        
        for (int dim = 0; dim < ndim_; ++dim) {
            MKL_INT n = dims_[dim];
            
            if (boundaries_[dim] == BoundaryType::ABSORBING) {
                // Precompute sine table for DST
                sin_tables_[dim].resize(n * n);
                for (MKL_INT k = 0; k < n; ++k) {
                    for (MKL_INT j = 0; j < n; ++j) {
                        sin_tables_[dim][k * n + j] = 
                            sin(M_PI * (j + 1) * (k + 1) / (n + 1));
                    }
                }
            } else if (boundaries_[dim] == BoundaryType::REFLECTING) {
                // Precompute cosine tables for DCT
                cos_tables_[dim].resize(n * n);
                for (MKL_INT k = 0; k < n; ++k) {
                    for (MKL_INT j = 0; j < n; ++j) {
                        cos_tables_[dim][k * n + j] = 
                            cos(M_PI * (j + 0.5) * k / n);
                    }
                }
            }
        }
    }
    
    void setupDescriptors() {
        fft_handles_.resize(ndim_, nullptr);
        
        for (int dim = 0; dim < ndim_; ++dim) {
            if (boundaries_[dim] == BoundaryType::PERIODIC) {
                setupPeriodicTransform(dim);
            }
        }
    }
    
    void setupPeriodicTransform(int dim) {
        MKL_LONG status;
        MKL_INT length = dims_[dim];
        
        // Use DFTI_COMPLEX for consistent R2R transforms
        status = DftiCreateDescriptor(&fft_handles_[dim], 
                                      DFTI_DOUBLE, 
                                      DFTI_COMPLEX, 
                                      1, 
                                      length);
        checkStatus(status, "DftiCreateDescriptor");
        
        status = DftiSetValue(fft_handles_[dim], 
                             DFTI_PLACEMENT, 
                             DFTI_INPLACE);
        checkStatus(status, "DftiSetValue PLACEMENT");
        
        status = DftiCommitDescriptor(fft_handles_[dim]);
        checkStatus(status, "DftiCommitDescriptor");
    }
    
    void applyForwardTransform(int dim) {
        if (boundaries_[dim] == BoundaryType::PERIODIC) {
            applyFFT_Optimized(dim, true);
        } else if (boundaries_[dim] == BoundaryType::ABSORBING) {
            applyDST_Optimized(dim);
        } else if (boundaries_[dim] == BoundaryType::REFLECTING) {
            applyDCT_Optimized(dim, true);
        }
    }
    
    void applyBackwardTransform(int dim) {
        if (boundaries_[dim] == BoundaryType::PERIODIC) {
            applyFFT_Optimized(dim, false);
        } else if (boundaries_[dim] == BoundaryType::ABSORBING) {
            applyDST_Optimized(dim); // Self-inverse
        } else if (boundaries_[dim] == BoundaryType::REFLECTING) {
            applyDCT_Optimized(dim, false);
        }
    }
    
    void applyFFT_Optimized(int dim, bool forward) {
        MKL_INT n = dims_[dim];
        MKL_INT stride = strides_[dim];
        MKL_INT num_trans = num_transforms_[dim];
        
        std::vector<std::complex<double>> result_buffer(total_size_);
        double scale = 1.0 / sqrt(static_cast<double>(n));
        
        for (MKL_INT batch = 0; batch < num_trans; ++batch) {
            for (MKL_INT s = 0; s < stride; ++s) {
                MKL_INT offset = batch * n * stride + s;
                
                std::vector<MKL_Complex16> temp_complex(n);
                
                // Load data
                if (is_complex_) {
                    for (MKL_INT i = 0; i < n; ++i) {
                        temp_complex[i].real = complex_buffer_[offset + i * stride].real();
                        temp_complex[i].imag = complex_buffer_[offset + i * stride].imag();
                    }
                } else {
                    for (MKL_INT i = 0; i < n; ++i) {
                        temp_complex[i].real = real_buffer_[offset + i * stride];
                        temp_complex[i].imag = 0.0;
                    }
                }
                
                // Apply transform
                MKL_LONG status;
                if (forward) {
                    status = DftiComputeForward(fft_handles_[dim], temp_complex.data());
                } else {
                    status = DftiComputeBackward(fft_handles_[dim], temp_complex.data());
                }
                checkStatus(status, "DftiCompute");
                
                // Store with scaling
                for (MKL_INT i = 0; i < n; ++i) {
                    result_buffer[offset + i * stride] = 
                        std::complex<double>(temp_complex[i].real * scale, 
                                            temp_complex[i].imag * scale);
                }
            }
        }
        
        complex_buffer_ = result_buffer;
    }
    
    void convertComplexToReal() {
        for (MKL_INT i = 0; i < total_size_; ++i) {
            real_buffer_[i] = complex_buffer_[i].real();
        }
        is_complex_ = false;
    }
    
    void convertRealToComplex() {
        for (MKL_INT i = 0; i < total_size_; ++i) {
            complex_buffer_[i] = std::complex<double>(real_buffer_[i], 0.0);
        }
        is_complex_ = true;
    }
    
    void applyDST_Optimized(int dim) {
        MKL_INT n = dims_[dim];
        MKL_INT stride = strides_[dim];
        MKL_INT num_trans = num_transforms_[dim];
        
        // Temporary storage for results
        std::vector<double> result_buffer(total_size_);
        
        for (MKL_INT batch = 0; batch < num_trans; ++batch) {
            for (MKL_INT s = 0; s < stride; ++s) {
                MKL_INT offset = batch * n * stride + s;
                
                // Compute DST Type-I (orthogonal form)
                for (MKL_INT k = 0; k < n; ++k) {
                    double sum = 0.0;
                    double factor = sqrt(2.0 / (n + 1));
                    
                    for (MKL_INT j = 0; j < n; ++j) {
                        sum += real_buffer_[offset + j * stride] * 
                               sin(M_PI * (j + 1) * (k + 1) / (n + 1)) * factor;
                    }
                    
                    result_buffer[offset + k * stride] = sum;
                }
            }
        }
        
        // Copy result back
        real_buffer_ = result_buffer;
    }
    
    void applyDCT_Optimized(int dim, bool forward) {
        MKL_INT n = dims_[dim];
        MKL_INT stride = strides_[dim];
        MKL_INT num_trans = num_transforms_[dim];
        
        // Temporary storage for results
        std::vector<double> result_buffer(total_size_);
        
        for (MKL_INT batch = 0; batch < num_trans; ++batch) {
            for (MKL_INT s = 0; s < stride; ++s) {
                MKL_INT offset = batch * n * stride + s;
                
                if (forward) {
                    // DCT Type-II
                    for (MKL_INT k = 0; k < n; ++k) {
                        double sum = 0.0;
                        
                        for (MKL_INT j = 0; j < n; ++j) {
                            double factor = (k == 0) ? sqrt(1.0 / n) : sqrt(2.0 / n);
                            sum += real_buffer_[offset + j * stride] * 
                                   cos(M_PI * (j + 0.5) * k / n) * factor;
                        }
                        
                        result_buffer[offset + k * stride] = sum;
                    }
                } else {
                    // DCT Type-III (IDCT)
                    for (MKL_INT k = 0; k < n; ++k) {
                        double sum = 0.0;
                        
                        for (MKL_INT j = 0; j < n; ++j) {
                            double factor = (j == 0) ? sqrt(1.0 / n) : sqrt(2.0 / n);
                            sum += real_buffer_[offset + j * stride] * 
                                   cos(M_PI * j * (k + 0.5) / n) * factor;
                        }
                        
                        result_buffer[offset + k * stride] = sum;
                    }
                }
            }
        }
        
        // Copy result back
        real_buffer_ = result_buffer;
    }
    
    double getNormalization() const {
        double norm = 1.0;
        for (int dim = 0; dim < ndim_; ++dim) {
            switch(boundaries_[dim]) {
                case BoundaryType::PERIODIC:
                    norm *= dims_[dim];
                    break;
                case BoundaryType::ABSORBING:
                    norm *= (dims_[dim] + 1);
                    break;
                case BoundaryType::REFLECTING:
                    norm *= dims_[dim];
                    break;
            }
        }
        return norm;
    }
    
    void checkStatus(MKL_LONG status, const char* operation) {
        if (status != 0) {
            std::cerr << "MKL Error in " << operation 
                     << ": " << status << std::endl;
            throw std::runtime_error("MKL operation failed");
        }
    }
    
    void cleanup() {
        for (auto& handle : fft_handles_) {
            if (handle != nullptr) {
                DftiFreeDescriptor(&handle);
            }
        }
    }
};

// Timing utility
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// Compilation command:
/*
g++ -O3 -std=c++11 mixed_boundary_fft_mkl.cpp -o mixed_fft \
    -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
*/