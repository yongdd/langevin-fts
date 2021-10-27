%filename = 'temp/scft3d_fields_initial.mat';
filename = 'temp/scft3d_fields_final.mat';
%filename = 'temp/fields_1_000102.mat';

load(filename);

%Plot 3D
h=figure;
%v = reshape(phi_a, nx);
%v = reshape(w_minus(1:32^3), [32,32,32]);
v = reshape(w(1:32^3), [32,32,32]);
%v = permute(v,[3 2 1]);

[x,y,z] = meshgrid(1:32,1:32,1:32);
isovalue = -100.0;
%isovalue = 0.65;

p1 = patch(isosurface(x,y,z,v,isovalue));
isonormals(x,y,z,v,p1)
set(p1,'FaceColor','b','EdgeColor','none');

p2 = patch(isocaps(x,y,z,v,isovalue),'FaceColor','interp',...
    'EdgeColor','none');

%Options
%caxis([0.0 2.5]);
colormap jet
view(3);
axis off
lighting gouraud
axis([1 ysize 1 xsize 1 zsize])
set(gca,'XColor',[0 0 0],'XTick',[])
set(gca,'YColor',[0 0 0],'YTick',[])
set(gca,'ZColor',[0 0 0],'ZTick',[])

%Camera & Light
daspect([1 1 1])     % to change the ratio of axis
%daspect([width length height])
view(0,0)
%camlight left
%camlight(80,45)
view(60,30)
camlight right
%camlight left
%

set(h, 'PaperPositionMode', 'auto');
% [ auto | {manual} ]
set(h, 'PaperUnits', 'points');
% [ {inches} | centimeters | normalized | points ]
set(h, 'PaperPosition', [0 0 800 500]);
% [left,bottom,width,height]
filename = "out";
print (h,filename,'-dpng') % print (h,'bulk','-dpdf')
%close(h)
