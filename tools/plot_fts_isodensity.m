
% Load Data
load("data_simulation_chin18.0/fields_100000.mat");
v = reshape(phi_a,[nx(3), nx(2), nx(1)]);
v = permute(v,[2 3 1]);

% Colormap
h=figure;
c = jet(1024);
colormap(c)
caxis([0.5 1])

% Mesh
[x,y,z] = meshgrid(1:nx(1),1:nx(2),1:nx(3));
x = double(x); y = double(y); z = double(z);
%v = smooth3(v);
isovalue = 0.5;

% Isosurface
p1 = patch(isosurface(x,y,z,v,isovalue));
isonormals(x,y,z,v,p1)
set(p1,'FaceColor',c(1,:),'EdgeColor','none');
p2 = patch(isocaps(x,y,z,v,isovalue),'FaceColor','interp',...
    'EdgeColor','none');
alpha(p1,0.5)
%alpha(p2,0.8)

% View & Light
axis([1 nx(1) 1 nx(2) 1 nx(3)])
axis off
lighting gouraud
daspect(double(nx)./lx)     % to change the ratio of axis
view(30,20)
camlight right

% Save
set(h, 'PaperPositionMode', 'auto');     % [ auto | {manual} ]
set(h, 'PaperUnits', 'points');          % [ {inches} | centimeters | normalized | points ]
set(h, 'PaperPosition', [0 0 800 500]);  % [left,bottom,width,height]
% [~,outfilename,~] = fileparts(filename);
print (h,"isodensity",'-dpng') % print (h,'bulk','-dpdf')
% close(h)
