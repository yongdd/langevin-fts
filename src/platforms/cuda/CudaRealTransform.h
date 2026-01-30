/**
 * @file CudaRealTransform.h
 * @brief Unified CUDA Real Transform (DCT/DST) implementation.
 *
 * Supports DCT Types 1-4 and DST Types 1-4 for 1D, 2D, and 3D arrays.
 * Allows mixing different transform types across dimensions.
 *
 * Boundary condition mapping:
 *   - DCT (Discrete Cosine Transform): Neumann boundary conditions
 *   - DST (Discrete Sine Transform): Dirichlet boundary conditions
 *
 * FFTW correspondence:
 *   - DCT-1: FFTW_REDFT00   - DST-1: FFTW_RODFT00
 *   - DCT-2: FFTW_REDFT10   - DST-2: FFTW_RODFT10
 *   - DCT-3: FFTW_REDFT01   - DST-3: FFTW_RODFT01
 *   - DCT-4: FFTW_REDFT11   - DST-4: FFTW_RODFT11
 *
 * Reference: FCT/FST algorithm (Makhoul 1980, Martucci 1994)
 *            CUDA implementation: https://github.com/rmingming/cuHelmholtz
 */

#ifndef CUDA_REAL_TRANSFORM_H_
#define CUDA_REAL_TRANSFORM_H_

#include <cuda_runtime.h>
#include <cufft.h>

/**
 * Transform Type enumeration for DCT and DST.
 */
enum CudaTransformType {
    // DCT Types (Neumann boundary conditions)
    CUDA_DCT_1 = 0,   // FFTW_REDFT00: DCT-I
    CUDA_DCT_2 = 1,   // FFTW_REDFT10: DCT-II
    CUDA_DCT_3 = 2,   // FFTW_REDFT01: DCT-III
    CUDA_DCT_4 = 3,   // FFTW_REDFT11: DCT-IV
    // DST Types (Dirichlet boundary conditions)
    CUDA_DST_1 = 4,   // FFTW_RODFT00: DST-I
    CUDA_DST_2 = 5,   // FFTW_RODFT10: DST-II
    CUDA_DST_3 = 6,   // FFTW_RODFT01: DST-III
    CUDA_DST_4 = 7    // FFTW_RODFT11: DST-IV
};

/**
 * @brief Check if transform type is DCT
 */
inline bool isDCT(CudaTransformType type) {
    return type <= CUDA_DCT_4;
}

/**
 * @brief Check if transform type is DST
 */
inline bool isDST(CudaTransformType type) {
    return type >= CUDA_DST_1;
}

/**
 * @brief Check if transform type is Type-1 (special handling required)
 */
inline bool isType1(CudaTransformType type) {
    return type == CUDA_DCT_1 || type == CUDA_DST_1;
}

/**
 * @brief Get the name string for a transform type
 */
inline const char* getTransformName(CudaTransformType type) {
    switch (type) {
        case CUDA_DCT_1: return "DCT-1";
        case CUDA_DCT_2: return "DCT-2";
        case CUDA_DCT_3: return "DCT-3";
        case CUDA_DCT_4: return "DCT-4";
        case CUDA_DST_1: return "DST-1";
        case CUDA_DST_2: return "DST-2";
        case CUDA_DST_3: return "DST-3";
        case CUDA_DST_4: return "DST-4";
        default: return "Unknown";
    }
}

//==============================================================================
// CudaRealTransform1D
//==============================================================================

/**
 * @class CudaRealTransform1D
 * @brief CUDA 1D Real Transform (DCT/DST Types 1-4).
 *
 * For DCT-1/DST-1: Input size is N+1/N-1, output size is N+1/N-1.
 * For DCT-2/3/4 and DST-2/3/4: Input size is N, output size is N.
 */
class CudaRealTransform1D
{
private:
    int N_;                      ///< Transform size
    CudaTransformType type_;     ///< Transform type
    cufftHandle plan_;           ///< cuFFT plan
    double* d_work_;             ///< Work buffer
    double* d_x1_;               ///< Auxiliary buffer
    // Optional lookup tables for DCT2/DCT3/DCT4 and DST2/3/4
    double* dct2_sin_{nullptr};
    double* dct2_cos_{nullptr};
    double* dct3_sin_{nullptr};
    double* dct3_cos_{nullptr};
    double* dct4_sin_{nullptr};
    double* dct4_cos_{nullptr};
    cudaStream_t stream_{0};     ///< CUDA stream for execution
    bool initialized_;

public:
    /**
     * @brief Construct 1D real transformer.
     * @param N Transform size
     * @param type Transform type (DCT-1/2/3/4 or DST-1/2/3/4)
     */
    CudaRealTransform1D(int N, CudaTransformType type);
    ~CudaRealTransform1D();

    /**
     * @brief Execute transform (in-place).
     * @param d_data Device array
     */
    void execute(double* d_data);
    void execute(double* d_data, cudaStream_t stream);

    /**
     * @brief Set CUDA stream for execution.
     */
    void set_stream(cudaStream_t stream);

    /**
     * @brief Get the input/output size.
     */
    int get_size() const;

    /**
     * @brief Get transform type.
     */
    CudaTransformType get_type() const { return type_; }

    /**
     * @brief Get normalization factor for round-trip.
     */
    double get_normalization() const;
};

//==============================================================================
// CudaRealTransform2D
//==============================================================================

/**
 * @class CudaRealTransform2D
 * @brief CUDA 2D Real Transform (DCT/DST).
 *
 * Supports any combination of DCT-2/3/4 and DST-2/3/4 across dimensions.
 * Type-1 transforms (DCT-1 or DST-1) must use the same type in both dimensions.
 *
 * Example use cases:
 *   - X: DCT-2 (Neumann), Y: DST-2 (Dirichlet)
 *   - X: DST-3 (Dirichlet), Y: DCT-3 (Neumann)
 */
class CudaRealTransform2D
{
private:
    int Nx_, Ny_;                ///< Transform sizes
    int M_;                      ///< Total size (Nx * Ny)
    int M_padded_;               ///< Padded size for FFT
    CudaTransformType type_x_;   ///< Transform type for X dimension
    CudaTransformType type_y_;   ///< Transform type for Y dimension

    cufftHandle plan_x_;         ///< cuFFT plan for X dimension
    cufftHandle plan_y_;         ///< cuFFT plan for Y dimension

    double* d_work_;             ///< Work buffer
    double* d_temp_;             ///< Temporary buffer
    double* d_x1_;               ///< Auxiliary buffer

    // Optional lookup tables for DCT2/DCT3/DCT4 and DST2/3/4
    double* dct2_sin_x_{nullptr};
    double* dct2_cos_x_{nullptr};
    double* dct2_sin_y_{nullptr};
    double* dct2_cos_y_{nullptr};
    double* dct3_sin_x_{nullptr};
    double* dct3_cos_x_{nullptr};
    double* dct3_sin_y_{nullptr};
    double* dct3_cos_y_{nullptr};
    double* dct4_sin_x_{nullptr};
    double* dct4_cos_x_{nullptr};
    double* dct4_sin_y_{nullptr};
    double* dct4_cos_y_{nullptr};

    bool initialized_;
    cudaStream_t stream_{0};     ///< CUDA stream for execution

    void init();

    // Y dimension processing
    void executeY_DCT1(double* d_data);
    void executeY_DCT2(double* d_data);
    void executeY_DCT3(double* d_data);
    void executeY_DCT4(double* d_data);
    void executeY_DST1(double* d_data);
    void executeY_DST2(double* d_data);
    void executeY_DST3(double* d_data);
    void executeY_DST4(double* d_data);

    // X dimension processing
    void executeX_DCT1(double* d_data);
    void executeX_DCT2(double* d_data);
    void executeX_DCT3(double* d_data);
    void executeX_DCT4(double* d_data);
    void executeX_DST1(double* d_data);
    void executeX_DST2(double* d_data);
    void executeX_DST3(double* d_data);
    void executeX_DST4(double* d_data);

public:
    /**
     * @brief Construct 2D real transformer with same type in both dimensions.
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param type Transform type for both dimensions
     */
    CudaRealTransform2D(int Nx, int Ny, CudaTransformType type);

    /**
     * @brief Construct 2D real transformer with different types per dimension.
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param type_x Transform type for X dimension
     * @param type_y Transform type for Y dimension
     * @note Type-1 transforms cannot be mixed with other types
     */
    CudaRealTransform2D(int Nx, int Ny, CudaTransformType type_x, CudaTransformType type_y);

    ~CudaRealTransform2D();

    /**
     * @brief Execute 2D transform (in-place).
     * @param d_data Device array
     */
    void execute(double* d_data);
    void execute(double* d_data, cudaStream_t stream);

    /**
     * @brief Set CUDA stream for execution.
     */
    void set_stream(cudaStream_t stream);

    /**
     * @brief Get the input/output size.
     */
    int get_size() const { return M_; }

    /**
     * @brief Get dimensions for this transform.
     */
    void get_dims(int& nx, int& ny) const;

    /**
     * @brief Get transform types for each dimension.
     */
    void get_types(CudaTransformType& type_x, CudaTransformType& type_y) const;

    /**
     * @brief Get normalization factor for round-trip.
     */
    double get_normalization() const;
};

//==============================================================================
// CudaRealTransform3D
//==============================================================================

/**
 * @class CudaRealTransform3D
 * @brief CUDA 3D Real Transform (DCT/DST).
 *
 * Supports any combination of DCT-2/3/4 and DST-2/3/4 across dimensions.
 * Type-1 transforms (DCT-1 or DST-1) must use the same type in all dimensions.
 *
 * Example use cases:
 *   - X: DCT-2, Y: DCT-2, Z: DST-2 (Neumann in XY, Dirichlet in Z)
 *   - X: DST-2, Y: DST-2, Z: DCT-2 (Dirichlet in XY, Neumann in Z)
 */
class CudaRealTransform3D
{
private:
    int Nx_, Ny_, Nz_;           ///< Transform sizes
    int M_;                      ///< Total size
    int M_padded_;               ///< Padded size for FFT
    CudaTransformType type_x_;   ///< Transform type for X dimension
    CudaTransformType type_y_;   ///< Transform type for Y dimension
    CudaTransformType type_z_;   ///< Transform type for Z dimension

    cufftHandle plan_x_;         ///< cuFFT plan for X dimension
    cufftHandle plan_y_;         ///< cuFFT plan for Y dimension
    cufftHandle plan_z_;         ///< cuFFT plan for Z dimension

    double* d_work_;             ///< Work buffer
    double* d_temp_;             ///< Temporary buffer
    double* d_x1_;               ///< Auxiliary buffer

    // Optional lookup tables for DCT2/DCT3 (reduce sincos overhead)
    double* dct2_sin_x_{nullptr};
    double* dct2_cos_x_{nullptr};
    double* dct2_sin_y_{nullptr};
    double* dct2_cos_y_{nullptr};
    double* dct2_sin_z_{nullptr};
    double* dct2_cos_z_{nullptr};
    double* dct3_sin_x_{nullptr};
    double* dct3_cos_x_{nullptr};
    double* dct3_sin_y_{nullptr};
    double* dct3_cos_y_{nullptr};
    double* dct3_sin_z_{nullptr};
    double* dct3_cos_z_{nullptr};
    double* dct4_sin_x_{nullptr};
    double* dct4_cos_x_{nullptr};
    double* dct4_sin_y_{nullptr};
    double* dct4_cos_y_{nullptr};
    double* dct4_sin_z_{nullptr};
    double* dct4_cos_z_{nullptr};

    bool initialized_;
    cudaStream_t stream_{0};     ///< CUDA stream for execution

    void init();

    // Z dimension processing
    void executeZ_DCT1(double* d_data);
    void executeZ_DCT2(double* d_data);
    void executeZ_DCT3(double* d_data);
    void executeZ_DCT4(double* d_data);
    void executeZ_DST1(double* d_data);
    void executeZ_DST2(double* d_data);
    void executeZ_DST3(double* d_data);
    void executeZ_DST4(double* d_data);

    // Y dimension processing
    void executeY_DCT1(double* d_data);
    void executeY_DCT2(double* d_data);
    void executeY_DCT3(double* d_data);
    void executeY_DCT4(double* d_data);
    void executeY_DST1(double* d_data);
    void executeY_DST2(double* d_data);
    void executeY_DST3(double* d_data);
    void executeY_DST4(double* d_data);

    // X dimension processing
    void executeX_DCT1(double* d_data);
    void executeX_DCT2(double* d_data);
    void executeX_DCT3(double* d_data);
    void executeX_DCT4(double* d_data);
    void executeX_DST1(double* d_data);
    void executeX_DST2(double* d_data);
    void executeX_DST3(double* d_data);
    void executeX_DST4(double* d_data);

public:
    /**
     * @brief Construct 3D real transformer with same type in all dimensions.
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param Nz Size in Z
     * @param type Transform type for all dimensions
     */
    CudaRealTransform3D(int Nx, int Ny, int Nz, CudaTransformType type);

    /**
     * @brief Construct 3D real transformer with different types per dimension.
     * @param Nx Size in X
     * @param Ny Size in Y
     * @param Nz Size in Z
     * @param type_x Transform type for X dimension
     * @param type_y Transform type for Y dimension
     * @param type_z Transform type for Z dimension
     * @note Type-1 transforms cannot be mixed with other types
     */
    CudaRealTransform3D(int Nx, int Ny, int Nz,
                        CudaTransformType type_x, CudaTransformType type_y, CudaTransformType type_z);

    ~CudaRealTransform3D();

    /**
     * @brief Execute 3D transform (in-place).
     * @param d_data Device array
     */
    void execute(double* d_data);
    void execute(double* d_data, cudaStream_t stream);

    /**
     * @brief Execute Z-only DCT-2 (in-place).
     *
     * Runs the Z-dimension DCT-2 and copies the result back to d_data.
     */
    void execute_z_dct2(double* d_data);
    void execute_z_dct2(double* d_data, cudaStream_t stream);
    void execute_z_dct2_to_zmajor(double* d_data, double* d_out_zmajor);
    void execute_z_dct2_to_zmajor(double* d_data, double* d_out_zmajor, cudaStream_t stream);

    /**
     * @brief Execute Z-only DCT-3 (in-place).
     *
     * Runs the Z-dimension DCT-3 and copies the result back to d_data.
     */
    void execute_z_dct3(double* d_data);
    void execute_z_dct3(double* d_data, cudaStream_t stream);
    void execute_z_dct3_from_zmajor(double* d_in_zmajor, double* d_out_xyz);
    void execute_z_dct3_from_zmajor(double* d_in_zmajor, double* d_out_xyz, cudaStream_t stream);

    /**
     * @brief Set CUDA stream for execution.
     */
    void set_stream(cudaStream_t stream);

    /**
     * @brief Get the input/output size.
     */
    int get_size() const { return M_; }

    /**
     * @brief Get dimensions for this transform.
     */
    void get_dims(int& nx, int& ny, int& nz) const;

    /**
     * @brief Get transform types for each dimension.
     */
    void get_types(CudaTransformType& type_x, CudaTransformType& type_y, CudaTransformType& type_z) const;

    /**
     * @brief Get normalization factor for round-trip.
     */
    double get_normalization() const;
};

//==============================================================================
// Backward compatibility aliases (deprecated)
//==============================================================================

// Old enum names for backward compatibility
typedef CudaTransformType CudaDCTType;
typedef CudaTransformType CudaDSTType;
typedef CudaTransformType CudaMixedTransformType;

// Old enum value aliases
#define CUDA_TRANSFORM_DCT_1 CUDA_DCT_1
#define CUDA_TRANSFORM_DCT_2 CUDA_DCT_2
#define CUDA_TRANSFORM_DCT_3 CUDA_DCT_3
#define CUDA_TRANSFORM_DCT_4 CUDA_DCT_4
#define CUDA_TRANSFORM_DST_1 CUDA_DST_1
#define CUDA_TRANSFORM_DST_2 CUDA_DST_2
#define CUDA_TRANSFORM_DST_3 CUDA_DST_3
#define CUDA_TRANSFORM_DST_4 CUDA_DST_4

// Old class name aliases
typedef CudaRealTransform1D CudaDCT;
typedef CudaRealTransform1D CudaDST;
typedef CudaRealTransform2D CudaDCT2D;
typedef CudaRealTransform2D CudaDST2D;
typedef CudaRealTransform2D CudaMixedTransform2D;
typedef CudaRealTransform3D CudaDCT3D;
typedef CudaRealTransform3D CudaDST3D;
typedef CudaRealTransform3D CudaMixedTransform3D;

#endif // CUDA_REAL_TRANSFORM_H_
