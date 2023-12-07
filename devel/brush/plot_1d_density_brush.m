clear all;

load("fields.mat");

data = phi_A;
disp(max(max(max(data))))
disp(min(min(min(data))))
mean(phi_A)

figure(1);
% plot(phi_A+phi_B);
% hold on;
% plot(phi_target);
% hold on;
plot(q_mask);
hold off;

figure(2);
x = linspace(-1,11,nx);
plot(x, phi_A);
hold on;
x = linspace(0,2);
L = 2;
phi_SST = 3*(L^2-x.^2)/(2*L^3);
plot(x, phi_SST);
hold off;
xlim([0,3]);

figure(3);
plot(w_A);
% hold on;
% plot(w_B);
hold off;