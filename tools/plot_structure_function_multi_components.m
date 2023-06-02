clear all;

file_path = 'data_simulation/structure_function';
type_pair = "A,A"; % "A,A", "A,B", "B,B" 
load(strcat(file_path,'_002000.mat'));

% Calculate k (Fourier mode) sqaure 
k2         = zeros(nx(1),nx(2),floor(double(nx(3))/2)+1);
k2_mapping = zeros(nx(1),nx(2),floor(double(nx(3))/2)+1);
for i = 0:nx(1)-1
    for j = 0:nx(2)-1
        for k = 0:floor(double(nx(3))/2)
            k2(i+1,j+1,k+1) = round((double(i)/lx(1))^2 + (double(j)/lx(2))^2 + (double(k)/lx(3))^2,7);
        end
    end
end

% Remove duplicates and set mapping
k2_unique = unique(k2);
for i = 0:nx(1)-1
    for j = 0:nx(2)-1
        for k = 0:floor(double(nx(3))/2)
            idx = find(k2_unique == round((double(i)/lx(1))^2 + (double(j)/lx(2))^2 + (double(k)/lx(3))^2,7));
            k2_mapping(i+1,j+1,k+1) = idx;
        end
    end
end

% Read data and caculate averages
sf_mag   = zeros(size(k2_unique));
sf_count = zeros(size(k2_unique));
for langevin_iter = 1000:1000:2000
    file_name = strcat(file_path,sprintf('_%06d.mat', langevin_iter));
    fprintf('%s\n', file_name);
    load(file_name);

    v = getfield(structure_function, type_pair);
    for i = 0:nx(1)-1
        for j = 0:nx(2)-1
            for k = 0:floor(double(nx(3))/2)
                idx = k2_mapping(i+1,j+1,k+1);
                sf_mag(idx) = sf_mag(idx) + v(i+1,j+1,k+1);
                sf_count(idx) = sf_count(idx) + 1;
            end
        end
    end
end
x = sqrt(double(k2_unique))*2*pi;
y = sf_mag./sf_count;
y(1) = 0.0;

% Plot
h=figure;
semilogy(x,y);
xlim([2 14])
ylim([0.01 1000])

%plot(x,y);
%xlim([0 3])
%ylim([0.0 1])
