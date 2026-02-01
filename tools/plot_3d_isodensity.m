% plot_3d_isodensity.m
% Render 3D isodensity surfaces with lighting.
%
% Usage:
%   Place this script in the directory containing fields_*.mat and run.
%
% Input:  fields_*.mat (3D simulation output)
% Output: isodensity.png

clear all;

% Load data
load("fields_002000.mat");

% Get phi_A (handle different variable naming conventions)
if exist('phi_A','var')
    phi_a = phi_A;
elseif exist('phi','var')
    phi_a = phi.A;
end

% Reshape and permute for correct orientation
v = reshape(phi_a, [nx(3), nx(2), nx(1)]);
v = permute(v, [2 3 1]);

% Grid spacing
nx = double(nx);
dx = lx ./ nx;

% Create figure
h = figure;

% Colormap
c = jet(1024);
colormap(c);
caxis([0.0 1.0]);

% Create mesh grid
[x, y, z] = meshgrid(dx(1):dx(1):lx(1), dx(2):dx(2):lx(2), dx(3):dx(3):lx(3));

% Isovalue at mean density
isovalue = mean(phi_a);

% Draw isosurface
p1 = patch(isosurface(x, y, z, v, isovalue));
isonormals(x, y, z, v, p1);
set(p1, 'FaceColor', c(1,:), 'EdgeColor', 'none');

% Draw isocaps
p2 = patch(isocaps(x, y, z, v, isovalue), 'FaceColor', 'interp', 'EdgeColor', 'none');
alpha(p1, 0.5);

% View and lighting
axis([0 lx(1) 0 lx(2) 0 lx(3)]);
lighting gouraud;
daspect([1 1 1]);
view(30, 20);
camlight right;

% Save figure
set(h, 'PaperPositionMode', 'auto');
set(h, 'PaperUnits', 'points');
set(h, 'PaperPosition', [0 0 800 500]);
print(h, 'isodensity', '-dpng');

fprintf("Saved to isodensity.png\n");
