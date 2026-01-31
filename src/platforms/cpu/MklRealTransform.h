/**
 * @file MklRealTransform.h
 * @brief MKL real-to-real DCT/DST (types 1-4) for 1D/2D/3D arrays.
 *
 * This provides a CPU counterpart to CudaRealTransform for internal use.
 * The transforms are unnormalized (FFTW-style); use get_normalization()
 * to obtain the round-trip scale factor.
 *
 * Implementation:
 *   - DCT-II, DCT-III, DCT-IV, DST-IV: MKL TT (Trigonometric Transform) interface
 *   - DCT-I, DST-I, DST-II, DST-III: Direct O(N²) computation (no MKL TT equivalent)
 *
 * MKL TT to FFTW mapping (empirically verified):
 *   - STAGGERED_COSINE backward × 2 = DCT-II
 *   - STAGGERED_COSINE forward × N  = DCT-III
 *   - STAGGERED2_COSINE forward × N = DCT-IV
 *   - STAGGERED2_SINE forward × N   = DST-IV
 */

#ifndef MKL_REAL_TRANSFORM_H_
#define MKL_REAL_TRANSFORM_H_

#include <array>
#include <memory>
#include "Exception.h"

class MklTTHandle;

/**
 * Transform Type enumeration for DCT and DST.
 *
 * FFTW correspondence:
 *   - DCT-1: FFTW_REDFT00   - DST-1: FFTW_RODFT00
 *   - DCT-2: FFTW_REDFT10   - DST-2: FFTW_RODFT10
 *   - DCT-3: FFTW_REDFT01   - DST-3: FFTW_RODFT01
 *   - DCT-4: FFTW_REDFT11   - DST-4: FFTW_RODFT11
 */
enum MklTransformType {
    MKL_DCT_1 = 0,
    MKL_DCT_2 = 1,
    MKL_DCT_3 = 2,
    MKL_DCT_4 = 3,
    MKL_DST_1 = 4,
    MKL_DST_2 = 5,
    MKL_DST_3 = 6,
    MKL_DST_4 = 7
};

inline bool isDCT(MklTransformType type) {
    return type <= MKL_DCT_4;
}

inline bool isDST(MklTransformType type) {
    return type >= MKL_DST_1;
}

inline bool isType1(MklTransformType type) {
    return type == MKL_DCT_1 || type == MKL_DST_1;
}

inline bool isType4(MklTransformType type) {
    return type == MKL_DCT_4 || type == MKL_DST_4;
}

inline const char* getTransformName(MklTransformType type) {
    switch (type) {
        case MKL_DCT_1: return "DCT-1";
        case MKL_DCT_2: return "DCT-2";
        case MKL_DCT_3: return "DCT-3";
        case MKL_DCT_4: return "DCT-4";
        case MKL_DST_1: return "DST-1";
        case MKL_DST_2: return "DST-2";
        case MKL_DST_3: return "DST-3";
        case MKL_DST_4: return "DST-4";
        default: return "Unknown";
    }
}

//==============================================================================
// MklRealTransform1D
//==============================================================================

class MklRealTransform1D
{
public:
    MklRealTransform1D(int N, MklTransformType type);
    ~MklRealTransform1D();

    // Non-copyable
    MklRealTransform1D(const MklRealTransform1D&) = delete;
    MklRealTransform1D& operator=(const MklRealTransform1D&) = delete;

    void execute(double* data);

    int get_size() const { return N_; }
    MklTransformType get_type() const { return type_; }
    double get_normalization() const;

private:
    int N_;
    MklTransformType type_;
    bool initialized_{false};

    // MKL TT handle for DCT-2, DCT-3, DCT-4, DST-4 (others use direct computation)
    std::unique_ptr<MklTTHandle> tt_handle_;

    void init();
};

//==============================================================================
// MklRealTransform2D
//==============================================================================

class MklRealTransform2D
{
public:
    MklRealTransform2D(int Nx, int Ny, MklTransformType type);
    MklRealTransform2D(int Nx, int Ny, MklTransformType type_x, MklTransformType type_y);
    ~MklRealTransform2D();

    // Non-copyable
    MklRealTransform2D(const MklRealTransform2D&) = delete;
    MklRealTransform2D& operator=(const MklRealTransform2D&) = delete;

    void execute(double* data);

    void get_dims(int& nx, int& ny) const { nx = nx_[0]; ny = nx_[1]; }
    void get_types(MklTransformType& type_x, MklTransformType& type_y) const {
        type_x = type_x_;
        type_y = type_y_;
    }
    double get_normalization() const;

private:
    std::array<int, 2> nx_;
    MklTransformType type_x_;
    MklTransformType type_y_;
    bool initialized_{false};

    std::unique_ptr<MklRealTransform1D> transform_x_;
    std::unique_ptr<MklRealTransform1D> transform_y_;

    void init();
    void getStrides(int dim, int& stride, int& num_transforms) const;
    void apply1D(double* data, int dim, MklRealTransform1D* transform) const;
};

//==============================================================================
// MklRealTransform3D
//==============================================================================

class MklRealTransform3D
{
public:
    MklRealTransform3D(int Nx, int Ny, int Nz, MklTransformType type);
    MklRealTransform3D(int Nx, int Ny, int Nz,
                       MklTransformType type_x,
                       MklTransformType type_y,
                       MklTransformType type_z);
    ~MklRealTransform3D();

    // Non-copyable
    MklRealTransform3D(const MklRealTransform3D&) = delete;
    MklRealTransform3D& operator=(const MklRealTransform3D&) = delete;

    void execute(double* data);

    void get_dims(int& nx, int& ny, int& nz) const { nx = nx_[0]; ny = nx_[1]; nz = nx_[2]; }
    void get_types(MklTransformType& type_x, MklTransformType& type_y, MklTransformType& type_z) const {
        type_x = type_x_;
        type_y = type_y_;
        type_z = type_z_;
    }
    double get_normalization() const;

private:
    std::array<int, 3> nx_;
    MklTransformType type_x_;
    MklTransformType type_y_;
    MklTransformType type_z_;
    bool initialized_{false};

    std::unique_ptr<MklRealTransform1D> transform_x_;
    std::unique_ptr<MklRealTransform1D> transform_y_;
    std::unique_ptr<MklRealTransform1D> transform_z_;

    void init();
    void getStrides(int dim, int& stride, int& num_transforms) const;
    void apply1D(double* data, int dim, MklRealTransform1D* transform) const;
};

// Type aliases for convenience
typedef MklRealTransform1D MklDCT;
typedef MklRealTransform1D MklDST;
typedef MklRealTransform2D MklDCT2D;
typedef MklRealTransform2D MklDST2D;
typedef MklRealTransform2D MklMixedTransform2D;
typedef MklRealTransform3D MklDCT3D;
typedef MklRealTransform3D MklDST3D;
typedef MklRealTransform3D MklMixedTransform3D;

#endif  // MKL_REAL_TRANSFORM_H_
