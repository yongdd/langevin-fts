% renormalization_discrete.m
% Compute the UV renormalization factor z_inf for the discrete chain model.
%
% The effective chi*N is: chi_N_eff = z_inf * chi_N
%
% References:
%   T. M. Beardsley and M. W. Matsen,
%   "Calibration of the Flory-Huggins interaction parameter in field-theoretic simulations,"
%   J. Chem. Phys. 150, 174902 (2019).
%
%   T. M. Beardsley and M. W. Matsen,
%   "Fluctuation correction for the order-disorder transition of diblock copolymer melts,"
%   J. Chem. Phys. 154, 124902 (2021).
%
% Usage:
%   Place this script in the directory containing fields_*.mat and run.
%
% Requirements:
%   - Discrete chain model
%   - Conformationally symmetric chains (epsilon = 1)
%
% The original version is written by T. M. Beardsley and M. W. Matsen.

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

%% Compute z_inf
dx = lx ./ double(nx);
dv = dx(1) * dx(2) * dx(3);
bond_t = 100;  % Number of bond terms to sum

% v_cell * rho_0
vcellrho = double(nbar)^0.5 * double(N) * dv;

% Summation of P_i using discrete chain
sum_p = 0.0;
for i = 1:bond_t
    sum_p = sum_p + p(i, dim, dx, double(N));
end

% Additional contribution using continuous chain approximation for tail
sum_p = sum_p + 2 / sqrt(0.5 + bond_t) * (3 * double(N) / (2 * pi))^1.5 * dv;

% Compute z_inf
z_inf = 1 - (1 + 2 * sum_p) / vcellrho;

fprintf("z_inf: %.7f\n", z_inf);

%% Helper function for discrete chain bond contribution
function output = p(i, dim, dx, N)
    output = 1;
    for idx = 1:dim
        output = output * dx(idx) * N^0.5 * (3.0 / (2.0 * pi * i))^0.5 * ...
                 erf(pi / dx(idx) / N^0.5 * (i / 6.0)^0.5);
    end
end
