clear all;

load("fields_000200.mat");
dx = lx./double(nx);
dv = dx(1)*dx(2)*dx(3);

if abs(epsilon - 1.0) > 1e-7
	fprintf("Currently, only conformationally symmetric chains (epsilon==1) are supported.\n");
	return;
end

if strcmpi(chain_model,'Discrete')
    fprintf("Use 'renormalization_discrete.m' for the discrete chain model.\n");
end

func = @(x,y,z) stucture_function_RPA_athermal(f, sqrt(x.^2+y.^2+z.^2)/sqrt(6));
int = integral3(func, -pi/dx(1), pi/dx(1), -pi/dx(2), pi/dx(2), -pi/dx(3), pi/dx(3));
z_inf = 1 - int/sqrt(double(nbar))/(8*pi^3*f*(1-f));

fprintf("z_inf: %.7f \n", z_inf);

% kernel function for linear AB diblock
function output = g(f,x)
output = 2*(f*x + exp(-f*x)-1)./x.^2;
end

% stucture_function_RPA with chiN = 0.0
function output = stucture_function_RPA_athermal(f,x)
x = x.^2;
output = (g(f,x).*g(1-f,x) - 0.25.*(g(1,x)-g(f,x)-g(1-f,x)).^2)./g(1,x);
end
