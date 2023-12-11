clear all;

load("fields.mat");

data = phi_A;
disp(max(max(max(data))))
disp(min(min(min(data))))
mean(phi_A)

x = linspace(-1,11,nx);

% figure(1);
% plot(phi_A+phi_B);
% hold on;
% plot(phi_target);
% hold on;
% plot(mask);
% hold off;

figure(2);
plot(x, phi_A);
hold on;
x_sst = linspace(0,2);
L = 2;
phi_SST = 3*(L^2-x_sst.^2)/(2*L^3);
plot(x_sst, phi_SST);
hold off;
% xlim([0,6]);

figure(3);
plot(x, w_A);
% hold on;
% plot(w_B);
hold off;
% xlim([0,6]);