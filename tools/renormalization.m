% This script is only valid only for conformationally symmetric diblock
clear all;

load("fields_000200.mat");
dx = lx./double(nx);
dv = dx(1)*dx(2)*dx(3);
bond_t = 100;

if strcmpi(chain_model,'Discrete')
    % calculate v_cell * rho_zero
    vcellrho = double(n_bar)^0.5*double(N)*dv;
    % summation of P_i using discrete chain
    sum = 0.0;
    for i=1:bond_t
        sum = sum + p(i, dim, dx, double(N));
    end
    % additional contribution is calculated using continuous chain
    sum = sum +  2/sqrt(0.5+bond_t)*(3*double(N)/(2*pi))^1.5*dv;
    z_inf = 1 - (1 + 2*sum)/vcellrho;
elseif strcmpi(chain_model,'Gaussian')
    func = @(x,y,z) stucture_function_RPA_athermal(f, sqrt(x.^2+y.^2+z.^2)/sqrt(6));
    int = integral3(func, -pi/dx(1), pi/dx(1), -pi/dx(2), pi/dx(2), -pi/dx(3), pi/dx(3));
    z_inf = 1 - int/sqrt(double(n_bar))/(8*pi^3*f*(1-f));
end

fprintf("z_inf: %.7f \n", z_inf);

% functions for discrete chain
function output = p(i, dim, dx, N)
output = 1;
for idx=1:dim
    output = output * dx(idx) * N^0.5 * (3.0/(2.0*pi*i))^0.5 * erf(pi/dx(idx)/N^0.5 *(i/6.0)^0.5);
end
end

% kernel function for linear AB diblock
function output = g(f,x)
output = 2*(f*x + exp(-f*x)-1)./x.^2;
end

% stucture_function_RPA with chiN = 0.0
function output = stucture_function_RPA_athermal(f,x)
x = x.^2;
output = (g(f,x).*g(1-f,x) - 0.25.*(g(1,x)-g(f,x)-g(1-f,x)).^2)./g(1,x);
end