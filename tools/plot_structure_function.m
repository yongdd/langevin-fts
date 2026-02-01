% plot_structure_function.m
% Plot spherically-averaged structure function S(k) from L-FTS simulation.
%
% Usage:
%   1. Edit file_path and type_pair below
%   2. Run in directory containing structure function files
%
% Input:  data_simulation/structure_function_*.mat
% Output: Figure displayed (save manually if needed)

clear all;

%% User parameters
file_path = 'data_simulation/structure_function';
type_pair = "A_A";  % Options: "A_A", "A_B", "B_B"
iter_start = 100000;
iter_end = 500000;
iter_step = 100000;

%% Load first file to get grid dimensions
load(strcat(file_path, '_100000.mat'));

%% Calculate k^2 for all Fourier modes
k2 = zeros(nx(1), nx(2), floor(double(nx(3))/2)+1);
for i = 0:nx(1)-1
    for j = 0:nx(2)-1
        for k = 0:floor(double(nx(3))/2)
            ki = min(i, nx(1)-i);
            kj = min(j, nx(2)-j);
            kk = k;
            k2(i+1, j+1, k+1) = round((double(ki)/lx(1))^2 + ...
                                       (double(kj)/lx(2))^2 + ...
                                       (double(kk)/lx(3))^2, 7);
        end
    end
end

%% Create mapping from k^2 to unique index
k2_unique = unique(k2);
k2_mapping = zeros(size(k2));
for i = 0:nx(1)-1
    for j = 0:nx(2)-1
        for k = 0:floor(double(nx(3))/2)
            ki = min(i, nx(1)-i);
            kj = min(j, nx(2)-j);
            kk = k;
            val = round((double(ki)/lx(1))^2 + ...
                        (double(kj)/lx(2))^2 + ...
                        (double(kk)/lx(3))^2, 7);
            idx = find(k2_unique == val);
            k2_mapping(i+1, j+1, k+1) = idx;
        end
    end
end

%% Accumulate structure function values
sf_mag = zeros(size(k2_unique));
sf_count = zeros(size(k2_unique));

for langevin_iter = iter_start:iter_step:iter_end
    file_name = strcat(file_path, sprintf('_%06d.mat', langevin_iter));
    fprintf('Loading %s\n', file_name);
    load(file_name);

    v = eval(strcat("structure_function_", type_pair));
    for i = 0:nx(1)-1
        for j = 0:nx(2)-1
            for k = 0:floor(double(nx(3))/2)
                idx = k2_mapping(i+1, j+1, k+1);
                sf_mag(idx) = sf_mag(idx) + v(i+1, j+1, k+1);
                sf_count(idx) = sf_count(idx) + 1;
            end
        end
    end
end

%% Compute spherically-averaged S(k)
x = sqrt(double(k2_unique)) * 2 * pi;  % k = 2*pi*sqrt(k2)
y = sf_mag ./ sf_count;
y(1) = 0.0;  % Remove k=0 mode

%% Plot
h = figure;
semilogy(x, y, 'LineWidth', 1.5);
xlabel('$kR_0$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$S(k)\rho_0 N$', 'Interpreter', 'latex', 'FontSize', 14);
title(sprintf('Structure Function (%s)', type_pair), 'FontSize', 14);
xlim([2 14]);
ylim([0.01 1000]);
grid on;

fprintf("Plot complete.\n");
