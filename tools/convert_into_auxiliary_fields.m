% convert_into_auxiliary_fields.m
% Convert monomer potential fields (w_A, w_B, ...) to auxiliary potential fields.
%
% Reference:
%   D. Duchs, K. T. Delaney, and G. H. Fredrickson,
%   "A multi-species exchange model for fully fluctuating polymer field theory simulations,"
%   J. Chem. Phys. 141, 174103 (2014).
%
% Usage:
%   Place this script in the directory containing fields_*.mat and run.
%
% Input:  fields_*.mat (with w_A, w_B, ..., matrix_a_inverse, monomer_types)
% Output: w_aux.mat

clear all;

% Load data
load("fields_000200.mat");

% The number of chemically distinct monomer types
S = length(monomer_types);

% Total grid points
M = prod(nx);

% Initialize matrix for auxiliary potential fields
w_aux = zeros([S M]);

% Convert monomer potential fields into auxiliary potential fields
% w_aux = A^(-1) * w_monomer
for i = 1:S
    for j = 1:S
        w_aux(i,:) = w_aux(i,:) + matrix_a_inverse(i,j)*eval(strcat("w_", monomer_types(j)));
    end
end

% Save the auxiliary fields
save("w_aux.mat", "w_aux", "nx", "lx", "monomer_types")
fprintf("Saved auxiliary fields to w_aux.mat\n");
