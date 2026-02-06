% Plot WTMD Statistics
%
% Usage:
%   1. Copy this script to the directory containing wtmd_statistics_*.mat
%   2. Modify the filename below as needed
%   3. Run in MATLAB
%
% Reference:
%   T. M. Beardsley and M. W. Matsen, J. Chem. Phys. 157, 114902 (2022).

clear all;

% Load WTMD statistics file
load("wtmd_statistics_1000000.mat");

% Plot U(Ψ) - Bias potential
figure(1);
plot(psi_range, u);
xlabel('\Psi')
ylabel('U(\Psi)')
title('Bias Potential')
grid on;

% Plot U'(Ψ) - Derivative of bias potential
figure(2);
plot(psi_range, up);
xlabel('\Psi')
ylabel('U^\prime(\Psi)')
title('Bias Potential Derivative')
grid on;

% Plot P(Ψ) - Probability distribution
figure(3);
coeff_t = 1/dT + 1;
exp_u = exp(u * coeff_t);
y = exp_u / sum(exp_u) / dpsi;

plot(psi_range, y);
xlabel('\Psi')
ylabel('P(\Psi)')
title('Probability Distribution')
grid on;

% Plot dF/dχN - Free energy derivative
figure(4);
threshold = 1e-1;
I0_norm = I0 / max(I0);
x = psi_range(I0_norm > threshold);
y = dH_psi_A_B(I0_norm > threshold);

plot(x, y);
xlabel('\Psi')
ylabel('\partial F/\partial{\chi_{AB} N}')
title('Free Energy Derivative')
grid on;
