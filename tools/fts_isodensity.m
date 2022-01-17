
% Read Data
load("fields_001000.mat");
h=figure;
v = reshape(phi_a,[nx(3), nx(2), nx(1)]);
v = permute(v,[2 3 1]);

% Isodensity Plot
isovalue = 0.5;
[x,y,z] = meshgrid(1:nx(1),1:nx(2),1:nx(3));
x = double(x);
y = double(y);
z = double(z); 
%v = smooth3(v);
p1 = patch(isosurface(x,y,z,v,isovalue));
isonormals(x,y,z,v,p1)
set(p1,'FaceColor','b','EdgeColor','none');
p2 = patch(isocaps(x,y,z,v,isovalue),'FaceColor','interp',...
    'EdgeColor','none');

% Options
colormap jet
view(3);
axis([1 nx(1) 1 nx(2) 1 nx(3)])
axis off

% Camera & Light
lighting gouraud
daspect(double(nx)./lx)     % to change the ratio of axis
view(0,0)
view(30,20)
camlight right

% Save image
set(h, 'PaperPositionMode', 'auto');     % [ auto | {manual} ]
set(h, 'PaperUnits', 'points');          % [ {inches} | centimeters | normalized | points ]
set(h, 'PaperPosition', [0 0 800 500]);  % [left,bottom,width,height]
print (h,"isodensity",'-dpng') % print (h,'bulk','-dpdf')
% close(h)
