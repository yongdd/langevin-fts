/**
 * @file FiniteDifference.cpp
 * @brief Implementation of finite difference coefficients.
 *
 * Provides utility functions for computing tridiagonal matrix coefficients
 * used in the Crank-Nicolson real-space solver. The matrices represent
 * the discretized Laplacian operator with various boundary conditions.
 *
 * **Crank-Nicolson Discretization:**
 *
 * The diffusion equation ∂q/∂s = (b²/6)∇²q is discretized as:
 *     (I - ds/2 · L) q^(n+1) = (I + ds/2 · L) q^n
 *
 * where L is the discrete Laplacian with coefficients:
 *     L[i,i-1] = L[i,i+1] = factor = b²ds/(6dx²)
 *     L[i,i] = -2·factor
 *
 * **Cell-Centered Grid:**
 *
 * Grid points are at cell centers: x_i = (i + 0.5) * dx for i = 0, 1, ..., N-1.
 * Boundaries are at x = 0 and x = L (cell faces, not grid points).
 *
 * **Boundary Condition Modifications (using ghost cells):**
 *
 * - PERIODIC: No modification (handled by cyclic tridiagonal solver)
 * - REFLECTING: Symmetric ghost (q_{-1} = q_0) → diagonal += off-diagonal
 * - ABSORBING: Antisymmetric ghost (q_{-1} = -q_0) → diagonal -= off-diagonal
 *
 * The antisymmetric ghost for absorbing BC enforces q = 0 at the cell face
 * (halfway between ghost cell and first cell).
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include "FiniteDifference.h"

/**
 * @brief Compute tridiagonal Laplacian matrix coefficients.
 *
 * Fills arrays with coefficients for each spatial direction,
 * applying boundary condition modifications at domain edges.
 *
 * @param bc             Boundary conditions [xl, xh, yl, yh, zl, zh]
 * @param nx             Grid points per dimension
 * @param dx             Grid spacing per dimension
 * @param xl,xd,xh       X-direction: lower, diagonal, upper (size nx[0])
 * @param yl,yd,yh       Y-direction coefficients (size nx[1])
 * @param zl,zd,zh       Z-direction coefficients (size nx[2])
 * @param bond_length_sq Statistical segment length squared (b²)
 * @param ds             Contour step size
 */
void FiniteDifference::get_laplacian_matrix(
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx,
        double *xl, double *xd, double *xh,
        double *yl, double *yd, double *yh,
        double *zl, double *zd, double *zh,
        double bond_length_sq, double ds)
{
    double xfactor[3] = {0.0};

    const int DIM = nx.size();

    // Calculate the exponential factor
    for(int d=0; d<DIM; d++)
        xfactor[d] = bond_length_sq*ds/(std::pow(dx[d],2)*6.0);

    if(DIM >= 1)
    {
        for(int i=0; i<nx[0]; i++)
        {
            xl[i] = -0.5*xfactor[0];
            xd[i] =  1.0+xfactor[0];
            xh[i] = -0.5*xfactor[0];
        }
    }
    if(DIM >= 2)
    {
        for(int i=0; i<nx[1]; i++)
        {
            yl[i] = -0.5*xfactor[1];
            yd[i] =  1.0+xfactor[1];
            yh[i] = -0.5*xfactor[1];
        }
    }
    if(DIM >= 3)
    {
        for(int i=0; i<nx[2]; i++)
        {
            zl[i] = -0.5*xfactor[2];
            zd[i] =  1.0+xfactor[2];
            zh[i] = -0.5*xfactor[2];
        }
    }

    for(int d=0; d<DIM; d++)
    {
        const int _nx_max = nx[d]-1;
        BoundaryCondition bcl = bc[2*d];
        BoundaryCondition bch = bc[2*d+1];
        double *_xl;
        double *_xd;
        double *_xh;

        if(d == 0)
        {
            _xl = xl;
            _xd = xd;
            _xh = xh;
        }
        else if(d == 1)
        {
            _xl = yl;
            _xd = yd;
            _xh = yh;
        }
        else if(d == 2)
        {
            _xl = zl;
            _xd = zd;
            _xh = zh;
        }

        // Cell-centered boundary modifications using ghost cells:
        // - Reflecting: symmetric ghost (q_{-1} = q_0)
        //   L*q_0 = (q_{-1} - 2*q_0 + q_1)/dx² = (-q_0 + q_1)/dx²
        //   → diagonal coefficient reduced by factor of off-diagonal
        // - Absorbing: antisymmetric ghost (q_{-1} = -q_0)
        //   L*q_0 = (-q_0 - 2*q_0 + q_1)/dx² = (-3*q_0 + q_1)/dx²
        //   → diagonal coefficient increased by factor of off-diagonal
        if(bcl == BoundaryCondition::REFLECTING)
        {
            _xd[0] += _xl[0];  // Symmetric ghost: subtract |xl| from diagonal
            _xl[0] = 0.0;
        }
        else if(bcl == BoundaryCondition::ABSORBING)
        {
            _xd[0] -= _xl[0];  // Antisymmetric ghost: add |xl| to diagonal
            _xl[0] = 0.0;
        }

        if(bch == BoundaryCondition::REFLECTING)
        {
            _xd[_nx_max] += _xh[_nx_max];  // Symmetric ghost
            _xh[_nx_max] = 0.0;
        }
        else if(bch == BoundaryCondition::ABSORBING)
        {
            _xd[_nx_max] -= _xh[_nx_max];  // Antisymmetric ghost
            _xh[_nx_max] = 0.0;
        }
    }
}