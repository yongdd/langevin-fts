clear all;

%Load File
% load("04828/fields.mat");
load("fields.mat");

%Plot
% disp(max(max(max(data))))
% disp(min(min(min(data))))
% data = reshape(data,[nx(3), nx(2), nx(1)]);

% phi_a_2d = reshape(data(:,:,16),[nx(2), nx(3)]);
% figure(1);
% data = reshape(q_mask,[nx(3), nx(2), nx(1)]);
% data = reshape(data(:,:,150),[nx(2), nx(1)]);
% data = permute(data,[2 1]);
% image(data,'CDataMapping','scaled');
% set(gca,'DataAspectRatio',[1 1 1])

L_mask = 1.0;
radius = 1.0;
dis_from_sub = 0.1;
dx = lx./double(nx);
range_x = round(L_mask/dx(1)+1):round(nx(2)*0.4);
range_y = round(nx(2)*0.2)+1:round(nx(2)*0.8);
[Y,X] = meshgrid(range_y,range_x);
Y = double(Y)*dx(2) - lx(2)/2-dx(2);
X = double(X)*dx(1)-L_mask-dx(1);

figure(2);
data = reshape(phi_A,[nx(3), nx(2), nx(1)]);
data = reshape(data(nx(3)/2,range_y,range_x),[length(range_y), length(range_x)]); %,111:190,31:100),[80, 70]);
data = permute(data,[2 1]);
contour(Y,X, data, [0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7], "ShowText", true, 'LineWidth', 1.0);
% image(data,'CDataMapping','scaled');
% c = jet(1024);
% colormap(flipud(c))
set(gca,'DataAspectRatio',[1 1 1])
hold on;

% Draw nano particle
% p = nsidedpoly(1000, 'Center', [dx(1) radius+dis_from_sub], 'Radius', radius-dx(1));
data = reshape(mask,[nx(3), nx(2), nx(1)]);
data = reshape(data(nx(3)/2,range_y,range_x),[length(range_y), length(range_x)]); %,111:190,31:100),[80, 70]);
data = permute(1.0-data,[2 1]);
contourf(Y,X, data, [0.999999 0.999999], "ShowText", false, 'LineWidth', 0.1);

% figure(3);
% data = reshape(w_A,[nx(3), nx(2), nx(1)]);
% data = reshape(data(150,:,:),[nx(2), nx(1)]);
% image(data,'CDataMapping','scaled');
% set(gca,'DataAspectRatio',[1 1 1])
% hold off;