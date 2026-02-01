% renormalization_rpa.m
% Compute the UV renormalization factor z_inf for the continuous chain model.
%
% The effective chi*N is: chi_N_eff = z_inf * chi_N
%
% Reference:
%   B. Vorselaars, P. Stasiak, and M. W. Matsen,
%   "Field-Theoretic Simulation of Block Copolymers at Experimentally Relevant Molecular Weights,"
%   Macromolecules 48, 9071 (2015).
%
% Usage:
%   Place this script in the directory containing fields_*.mat and run.
%
% Requirements:
%   - Continuous chain model
%   - Conformationally symmetric chains (epsilon = 1)

clear all;

%% Load data
load("fields_000200.mat");

%% Extract parameters
if exist('ds', 'var')
    N = 1.0 / ds;
end

% Get volume fraction f from density
if exist('phi_a', 'var')
    f = mean(phi_a);
elseif exist('phi_A', 'var')
    f = mean(phi_A);
elseif exist('phi', 'var')
    f = mean(phi.A);
end

% Get conformational asymmetry
if ~exist('epsilon', 'var')
    epsilon = initial_params.segment_lengths.A / initial_params.segment_lengths.B;
end

%% Check requirements
if abs(epsilon - 1.0) > 1e-7
    fprintf("Currently, only conformationally symmetric chains (epsilon==1) are supported.\n");
    return;
end

if strcmpi(chain_model, 'Discrete')
    fprintf("Use 'renormalization_discrete.m' for the discrete chain model.\n");
end

%% Compute z_inf using RPA structure function integral
dx = lx ./ double(nx);
dv = dx(1) * dx(2) * dx(3);

% Integrate RPA structure function over Brillouin zone
func = @(x, y, z) structure_function_RPA_athermal(f, sqrt(x.^2 + y.^2 + z.^2) / sqrt(6));
int_value = integral3(func, -pi/dx(1), pi/dx(1), -pi/dx(2), pi/dx(2), -pi/dx(3), pi/dx(3));

% Compute z_inf
z_inf = 1 - int_value / sqrt(double(nbar)) / (8 * pi^3 * f * (1 - f));

fprintf("z_inf: %.7f\n", z_inf);

%% Helper functions

% Debye function g(f, x) for linear chain
function output = g(f, x)
    output = 2 * (f * x + exp(-f * x) - 1) ./ x.^2;
end

% RPA structure function at chi*N = 0 (athermal)
function output = structure_function_RPA_athermal(f, x)
    x = x.^2;
    output = (g(f, x) .* g(1-f, x) - 0.25 * (g(1, x) - g(f, x) - g(1-f, x)).^2) ./ g(1, x);
end
