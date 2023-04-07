clear all;

load("fields_000200.mat");
if exist('ds','var')
    N = 1.0/ds;
end

dx = lx./double(nx);
dv = dx(1)*dx(2)*dx(3);
bond_t = 100;

if abs(epsilon - 1.0) > 1e-7
    fprintf("Currently, only conformationally symmetric chains (epsilon==1) are supported.\n");
    return;
end

// if ~strcmpi(chain_model,'Discrete')
//     fprintf("Your chain model is not the discrete chain model.\n");
// end

% calculate v_cell * rho_zero
vcellrho = double(nbar)^0.5*double(N)*dv;
% summation of P_i using discrete chain
sum = 0.0;
for i=1:bond_t
    sum = sum + p(i, dim, dx, double(N));
end
% additional contribution is calculated using continuous chain
sum = sum +  2/sqrt(0.5+bond_t)*(3*double(N)/(2*pi))^1.5*dv;
z_inf = 1 - (1 + 2*sum)/vcellrho;

fprintf("z_inf: %.7f \n", z_inf);

% functions for discrete chain
function output = p(i, dim, dx, N)
output = 1;
for idx=1:dim
    output = output * dx(idx) * N^0.5 * (3.0/(2.0*pi*i))^0.5 * erf(pi/dx(idx)/N^0.5 *(i/6.0)^0.5);
end
end
