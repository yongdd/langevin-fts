clear all;

% Load Data
load("fields_000200.mat");

% The number of chemically distinct monomer types
S = length(monomer_types);
% Grid number
M = prod(nx);

% Initialize matrix for exchange fields
w_exchange = zeros([S M]);

% Convert chemical species fields to multi-species exchange fields
for i = 1:S
    for j = 1:S
        w_exchange(i,:) = w_exchange(i,:) + matrix_a_inverse(i,j)*eval(strcat("w_", monomer_types(j)));
    end
end

% Save the exchange fields
save("w_exchange.mat")