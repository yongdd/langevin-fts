clear all;

load("fields.mat");

% phi_a_2d = reshape(data,[nx(2), nx(1)]);
% image(phi_a_2d(:,:),'CDataMapping','scaled');
% set(gca,'DataAspectRatio',[1 1 1]); % need to be changed...
% axis off;
% c = jet(1024);
% colormap(c)
% caxis([0.0 1])
% colorbar('FontSize',20)
% %caxis([-0.45 0.45])

% figure(1);
% data = reshape(phi_A+phi_B,[nx(2), nx(1)]);
% set(gca,'DataAspectRatio',[1 1 1])
% image(data,'CDataMapping','scaled');
% figure(2);
% data = reshape(phi_target,[nx(2), nx(1)]);
% image(data,'CDataMapping','scaled');
% set(gca,'DataAspectRatio',[1 1 1])
figure(3);
data = reshape(mask,[nx(2),nx(1)]);
image(data,'CDataMapping','scaled');
set(gca,'DataAspectRatio',[1 1 1])

figure(4);
data = reshape(phi_A,[nx(2), nx(1)]);
image(data,'CDataMapping','scaled');
set(gca,'DataAspectRatio',[1 1 1])
hold off;

% hold on;
% plot(phi_B);
% hold off;
% 
figure(5);
data = reshape(w_A,[nx(2), nx(1)]);
image(data,'CDataMapping','scaled');
set(gca,'DataAspectRatio',[1 1 1])
hold off;
