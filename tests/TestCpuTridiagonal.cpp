#include <iostream>
#include <cmath>
#include <numbers>
#include <algorithm>
#include "CpuSolverRealSpace.h"

int main()
{
    try
    {
        const int M = 10;
        double a[M] = {0.822383458999126,  0.180877073118435, 0.885692320279145,
                       0.89417010799111,   0.495074990864166, 0.612975741629382,
                       0.0415795198090432, 0.353431810889399, 0.773118461366249,
                       0.474587294381635};

        double b[M] = {0.200002276650706, 0.592127025743285, 0.460207620078036,
                       0.435945198378862, 0.61269588805607,  0.355979618324841,
                       0.548759176402544, 0.482897565408353, 0.541501788021353,
                       0.349682106604464};

        double c[M] = {0.104943438436657, 0.418086769592863, 0.61190542613784,
                       0.792961240687622, 0.713098832553561, 0.667410867822433,
                       0.492427261460169, 0.261956970404376, 0.479635996452285,
                       0.206215439022739};

        double y[M] = {0.180066106322814, 0.349989269840229, 0.580249533529743,
                       0.653060847246207, 0.793729416310673, 0.988032605316576,
                       0.98005550969782,  0.38678227079795,  0.894395839154923,
                       0.720491484453521};

        double x_answer[M] = {1.30521858221904, -0.771659313380314,  1.36532779447559,
                              1.0383434746621,  -1.28686656279112,   1.49787004664211,
                              1.86337898208832, -0.212765480943989, -0.645339592235395,
                              2.93627107620712};
        double x_periodic_answer[M] = {-37.7095597725768, -8.90673671271047, 29.7658392801646,
                                       -8.54639948044212, -28.0428600017516, 31.1409611959355,
                                        10.6262549404399, -12.4810831844606, 10.1475079039034,
                                        10.526436777743};

        double x[M] = {0.0};
        std::array<double,M> diff_sq;
        double error;

        // Tridiagonal
        CpuSolverRealSpace::tridiagonal(a, b, c, x, 1, y, M);
        std::cout << "Tridiagonal" << std::endl;
        std::cout << "i: x_answer[i], x[i]" << std::endl;
        for(int i=0; i<M; i++)
            std::cout << i << ": " << x_answer[i] << ", " << x[i] << std::endl;

        for(int i=0; i<M; i++)
            diff_sq[i] = pow(x_answer[i] - x[i],2);
        error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));

        std::cout<< "Inverse error: "<< error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
            return -1;

        // Tridiagonal with periodic boundary
        CpuSolverRealSpace::tridiagonal_periodic(a, b, c, x, 1, y, M);
        std::cout << "Tridiagonal with periodic boundary" << std::endl;
        std::cout << "i: x_periodic_answer[i], x[i]" << std::endl;
        for(int i=0; i<M; i++)
            std::cout << i << ": " << x_periodic_answer[i] << ", " << x[i] << std::endl;

        for(int i=0; i<M; i++)
            diff_sq[i] = pow(x_periodic_answer[i] - x[i],2);
        error = sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));

        std::cout<< "Inverse error: "<< error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
            return -1;

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
