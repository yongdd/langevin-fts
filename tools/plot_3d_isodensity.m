clear all;

% Load Data
load("fields_002000.mat");

% % Uncomment to read a json file
% json_str = fileread('fields.json');
% data = jsondecode(json_str);
% save('temp.mat','-struct','data')
% load("temp.mat");

if exist('phi_A','var')
    phi_a = phi_A;
elseif exist('phi','var')
    phi_a = phi.A;
end

v = reshape(phi_a,[nx(3), nx(2), nx(1)]);
v = permute(v,[2 3 1]);

nx = double(nx);
dx = lx./nx;

% Colormap
h=figure;
c = jet(1024);
colormap(c)
caxis([0.0 1])

% Mesh
[x,y,z] = meshgrid(dx(1):dx(1):lx(1),dx(2):dx(2):lx(2),dx(3):dx(3):lx(3));
%v = smooth3(v);
isovalue = mean(phi_a);

% Isosurface
p1 = patch(isosurface(x,y,z,v,isovalue));
isonormals(x,y,z,v,p1)
set(p1,'FaceColor',c(1,:),'EdgeColor','none');
p2 = patch(isocaps(x,y,z,v,isovalue),'FaceColor','interp',...
    'EdgeColor','none');
alpha(p1,0.5)

% View & Light
axis([0 lx(1) 0 lx(2) 0 lx(3)])
%axis off
lighting gouraud
daspect([1 1 1])
view(30,20)
camlight right

% Save
set(h, 'PaperPositionMode', 'auto');     % [ auto | {manual} ]
set(h, 'PaperUnits', 'points');          % [ {inches} | centimeters | normalized | points ]
set(h, 'PaperPosition', [0 0 800 500]);  % [left,bottom,width,height]
% [~,outfilename,~] = fileparts(filename);
print (h,'isodensity','-dpng') % print (h,'bulk','-dpdf')
% close(h)
