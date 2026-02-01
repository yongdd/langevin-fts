% plot_2d_density.m
% Plot 2D monomer density from simulation output.
%
% Usage:
%   Place this script in the directory containing fields.mat and run.
%
% Input:  fields.mat (2D simulation output)
% Output: 2d_density.png

clear all;

% Load data
load("fields.mat");

% Get phi_A (handle different variable naming conventions)
if exist('phi_A','var')
    phi_a = phi_A;
elseif exist('phi','var')
    phi_a = phi.A;
end

% Display min/max values
fprintf("phi_A min: %.4f, max: %.4f\n", min(phi_a(:)), max(phi_a(:)));

% Reshape to 2D grid
phi_a_2d = reshape(phi_a, [nx(2), nx(1)]);

% Plot
h = figure;
image(phi_a_2d, 'CDataMapping', 'scaled');
set(gca, 'DataAspectRatio', [lx(2) lx(1) 1]);
axis off;

% Colormap
colormap(jet(1024));
caxis([0.0 1.0]);
colorbar('FontSize', 20);

% Save figure
set(h, 'PaperPositionMode', 'auto');
set(h, 'PaperUnits', 'points');
set(h, 'PaperPosition', [0 0 500 500]);
print(h, "2d_density", '-dpng');

fprintf("Saved to 2d_density.png\n");
