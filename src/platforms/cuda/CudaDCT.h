/**
 * @file CudaDCT.h
 * @brief CUDA DCT (Discrete Cosine Transform) Types 1-4 implementation.
 *
 * Based on cuHelmholtz approach:
 *   - Pre-processing to convert DCT to FFT problem
 *   - cuFFT execution
 *   - Post-processing to reconstruct DCT coefficients
 *
 * Reference: https://github.com/rmingming/cuHelmholtz
 */

#ifndef CUDA_DCT_H_
#define CUDA_DCT_H_

#include <cuda_runtime.h>
#include <cufft.h>

/**
 * DCT Type enumeration matching FFTW conventions
 */
enum CudaDCTType {
    CUDA_DCT_1 = 0,  // FFTW_REDFT00: DCT-I
    CUDA_DCT_2 = 1,  // FFTW_REDFT10: DCT-II
    CUDA_DCT_3 = 2,  // FFTW_REDFT01: DCT-III (inverse of DCT-II)
    CUDA_DCT_4 = 3   // FFTW_REDFT11: DCT-IV
};

/**
 * @class CudaDCT
 * @brief CUDA DCT for 1D arrays (all types).
 *
 * Supports DCT types 1-4 matching FFTW's REDFT00/10/01/11.
 */
class CudaDCT
{
private:
    int N_;                  ///< Transform size
    CudaDCTType type_;       ///< DCT type

    cufftHandle plan_;       ///< cuFFT plan
    double* d_work_;         ///< Work buffer
    double* d_x1_;           ///< Auxiliary buffer for sum

    bool initialized_;

public:
    /**
     * @brief Construct 1D DCT transformer.
     *
     * @param N Transform size
     * @param type DCT type (1-4)
     */
    CudaDCT(int N, CudaDCTType type);
    ~CudaDCT();

    /**
     * @brief Execute DCT (in-place).
     * @param d_data Device array of size N (DCT-2,3,4) or N+1 (DCT-1)
     */
    void execute(double* d_data);

    /**
     * @brief Get the input/output size for this transform.
     */
    int get_size() const;

    /**
     * @brief Get normalization factor.
     * After forward + backward, multiply by this to get original.
     */
    double get_normalization() const;
};

/**
 * @class CudaDCT2D
 * @brief CUDA DCT for 2D arrays.
 *
 * Supports same type for both dimensions, or mixed types (DCT-2/3/4 only).
 * DCT-1 must use the same type for both dimensions.
 */
class CudaDCT2D
{
private:
    int Nx_, Ny_;            ///< Transform sizes
    CudaDCTType type_x_;     ///< DCT type for X dimension
    CudaDCTType type_y_;     ///< DCT type for Y dimension
    int M_;                  ///< Total size
    int M_padded_;           ///< Padded buffer size

    cufftHandle plan_x_;     ///< cuFFT plan for X dimension
    cufftHandle plan_y_;     ///< cuFFT plan for Y dimension

    double* d_work_;         ///< Work buffer
    double* d_temp_;         ///< Temporary buffer
    double* d_x1_;           ///< Auxiliary buffer

    bool initialized_;

    void init();

public:
    /**
     * @brief Construct 2D DCT transformer (same type for both dimensions).
     *
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param type DCT type (1-4)
     */
    CudaDCT2D(int Nx, int Ny, CudaDCTType type);

    /**
     * @brief Construct 2D DCT transformer (different types per dimension).
     *
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param type_x DCT type for X dimension (2, 3, or 4)
     * @param type_y DCT type for Y dimension (2, 3, or 4)
     * @note DCT-1 cannot be mixed with other types
     */
    CudaDCT2D(int Nx, int Ny, CudaDCTType type_x, CudaDCTType type_y);

    ~CudaDCT2D();

    /**
     * @brief Execute 2D DCT (in-place).
     * @param d_data Device array
     */
    void execute(double* d_data);

    /**
     * @brief Get the input/output size.
     */
    int get_size() const { return M_; }

    /**
     * @brief Get dimensions for this transform.
     */
    void get_dims(int& nx, int& ny) const;

    /**
     * @brief Get DCT types for each dimension.
     */
    void get_types(CudaDCTType& type_x, CudaDCTType& type_y) const;

    /**
     * @brief Get normalization factor for round-trip.
     */
    double get_normalization() const;
};

/**
 * @class CudaDCT3D
 * @brief CUDA DCT for 3D arrays.
 *
 * Supports same type for all dimensions, or mixed types (DCT-2/3/4 only).
 * DCT-1 must use the same type for all dimensions.
 */
class CudaDCT3D
{
private:
    int Nx_, Ny_, Nz_;       ///< Transform sizes
    CudaDCTType type_x_;     ///< DCT type for X dimension
    CudaDCTType type_y_;     ///< DCT type for Y dimension
    CudaDCTType type_z_;     ///< DCT type for Z dimension
    int M_;                  ///< Total size
    int M_padded_;           ///< Padded buffer size

    cufftHandle plan_x_;     ///< cuFFT plan for X dimension
    cufftHandle plan_y_;     ///< cuFFT plan for Y dimension
    cufftHandle plan_z_;     ///< cuFFT plan for Z dimension

    double* d_work_;         ///< Work buffer
    double* d_temp_;         ///< Temporary buffer
    double* d_x1_;           ///< Auxiliary buffer

    bool initialized_;

    void init();

public:
    /**
     * @brief Construct 3D DCT transformer (same type for all dimensions).
     *
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param Nz Size in Z
     * @param type DCT type (1-4)
     */
    CudaDCT3D(int Nx, int Ny, int Nz, CudaDCTType type);

    /**
     * @brief Construct 3D DCT transformer (different types per dimension).
     *
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param Nz Size in Z
     * @param type_x DCT type for X dimension (2, 3, or 4)
     * @param type_y DCT type for Y dimension (2, 3, or 4)
     * @param type_z DCT type for Z dimension (2, 3, or 4)
     * @note DCT-1 cannot be mixed with other types
     */
    CudaDCT3D(int Nx, int Ny, int Nz, CudaDCTType type_x, CudaDCTType type_y, CudaDCTType type_z);

    ~CudaDCT3D();

    /**
     * @brief Execute 3D DCT (in-place).
     * @param d_data Device array
     */
    void execute(double* d_data);

    /**
     * @brief Get the input/output size.
     */
    int get_size() const { return M_; }

    /**
     * @brief Get dimensions for this transform.
     */
    void get_dims(int& nx, int& ny, int& nz) const;

    /**
     * @brief Get DCT types for each dimension.
     */
    void get_types(CudaDCTType& type_x, CudaDCTType& type_y, CudaDCTType& type_z) const;

    /**
     * @brief Get normalization factor for round-trip.
     */
    double get_normalization() const;
};

#endif
