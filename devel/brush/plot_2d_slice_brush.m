clear all;

%Load File
load("fields_3d.mat");

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

figure(2);
data = reshape(phi_A,[nx(3), nx(2), nx(1)]);
data = reshape(data(150,111:190,31:100),[80, 70]);
data = permute(data,[2 1]);
contourf(data, "ShowText", true);
% image(data,'CDataMapping','scaled');
set(gca,'DataAspectRatio',[1 1 1])
hold off;

% figure(3);
% data = reshape(w_A,[nx(3), nx(2), nx(1)]);
% data = reshape(data(150,:,:),[nx(2), nx(1)]);
% image(data,'CDataMapping','scaled');
% set(gca,'DataAspectRatio',[1 1 1])
% hold off;