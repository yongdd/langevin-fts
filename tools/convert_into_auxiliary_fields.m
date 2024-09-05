clear all;

% Load Data
load("fields_000200.mat");

% The number of chemically distinct monomer types
S = length(monomer_types);
% Grid number
M = prod(nx);

% Initialize matrix for auxiliary fields
w_aux = zeros([S M]);

% Convert monomer chemical potential fields fields into auxiliary fields
for i = 1:S
    for j = 1:S
        w_aux(i,:) = w_aux(i,:) + matrix_a_inverse(i,j)*eval(strcat("w_", monomer_types(j)));
    end
end

% Save the auxiliary fields
save("w_aux.mat")