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
 * **Boundary Condition Modifications:**
 *
 * - PERIODIC: No modification (handled by cyclic tridiagonal solver)
 * - REFLECTING: Add off-diagonal to diagonal, zero off-diagonal
 * - ABSORBING: Zero off-diagonal only
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

        if(bcl == BoundaryCondition::REFLECTING)
        {
            _xd[0] += _xl[0];
            _xl[0] = 0.0;
        }
        else if(bcl == BoundaryCondition::ABSORBING)
        {
            _xl[0] = 0.0;
        }

        if(bch == BoundaryCondition::REFLECTING)
        {
            _xd[_nx_max] += _xh[_nx_max];
            _xh[_nx_max] = 0.0;
        }
        else if(bch == BoundaryCondition::ABSORBING)
        {
            _xh[_nx_max] = 0.0;
        }
    }
}

/**
 * @brief Construct Backward Euler Laplacian matrix coefficients.
 *
 * For Backward Euler: (I - ds * L) q^{n+1} = q^n
 * The matrix A = I - ds * D * ∇² where D = b²/6.
 *
 * With r = b² * ds / (6 * dx²):
 * - Off-diagonals: -r  (full step)
 * - Diagonal: 1 + 2*r
 */
void FiniteDifference::get_backward_euler_matrix(
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx,
        double *xl, double *xd, double *xh,
        double *yl, double *yd, double *yh,
        double *zl, double *zd, double *zh,
        double bond_length_sq, double ds)
{
    double xfactor[3] = {0.0};

    const int DIM = nx.size();

    // Calculate the factor r = b² * ds / (6 * dx²)
    for(int d=0; d<DIM; d++)
        xfactor[d] = bond_length_sq*ds/(std::pow(dx[d],2)*6.0);

    // Backward Euler uses full step (no 0.5 factor)
    if(DIM >= 1)
    {
        for(int i=0; i<nx[0]; i++)
        {
            xl[i] = -xfactor[0];           // Off-diagonal: -r (not -0.5*r)
            xd[i] =  1.0 + 2.0*xfactor[0]; // Diagonal: 1 + 2*r (not 1 + r)
            xh[i] = -xfactor[0];           // Off-diagonal: -r (not -0.5*r)
        }
    }
    if(DIM >= 2)
    {
        for(int i=0; i<nx[1]; i++)
        {
            yl[i] = -xfactor[1];
            yd[i] =  1.0 + 2.0*xfactor[1];
            yh[i] = -xfactor[1];
        }
    }
    if(DIM >= 3)
    {
        for(int i=0; i<nx[2]; i++)
        {
            zl[i] = -xfactor[2];
            zd[i] =  1.0 + 2.0*xfactor[2];
            zh[i] = -xfactor[2];
        }
    }

    // Apply boundary condition modifications (same as CN)
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

        if(bcl == BoundaryCondition::REFLECTING)
        {
            _xd[0] += _xl[0];
            _xl[0] = 0.0;
        }
        else if(bcl == BoundaryCondition::ABSORBING)
        {
            _xl[0] = 0.0;
        }

        if(bch == BoundaryCondition::REFLECTING)
        {
            _xd[_nx_max] += _xh[_nx_max];
            _xh[_nx_max] = 0.0;
        }
        else if(bch == BoundaryCondition::ABSORBING)
        {
            _xh[_nx_max] = 0.0;
        }
    }
}