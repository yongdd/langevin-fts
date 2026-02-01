% plot_2d_slice.m
% Extract and plot a 2D slice from 3D density data.
%
% Usage:
%   Place this script in the directory containing fields_*.mat and run.
%
% Input:  fields_*.mat (3D simulation output)
% Output: 2d_slice_image.png

clear all;

% Load data
load("fields_002000.mat");

% Get phi_A (handle different variable naming conventions)
if exist('phi_A','var')
    phi_a = phi_A;
elseif exist('phi','var')
    phi_a = phi.A;
end

% Display min/max values
fprintf("phi_A min: %.4f, max: %.4f\n", min(phi_a(:)), max(phi_a(:)));

% Reshape to 3D grid and extract first slice
data = reshape(phi_a, [nx(3), nx(2), nx(1)]);
phi_a_2d = reshape(data(:,:,1), [nx(2), nx(3)]);

% Plot
h = figure;
image(phi_a_2d, 'CDataMapping', 'scaled');
set(gca, 'DataAspectRatio', [1 1 1]);
axis off;

% Colormap
colormap(jet(1024));
caxis([0.0 1.0]);
colorbar('FontSize', 20);

% Save figure
set(h, 'PaperPositionMode', 'auto');
set(h, 'PaperUnits', 'points');
set(h, 'PaperPosition', [0 0 500 500]);
print(h, "2d_slice_image", '-dpng');

fprintf("Saved to 2d_slice_image.png\n");
