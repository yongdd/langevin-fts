clear all;

%Load File
load("fields.mat");
if exist('phi_A','var')
    phi_a = phi_A;
elseif exist('phi','var')
    phi_a = phi.A;
end
    
%Plot
h=figure;
%data = wpd;
data = phi_a;
disp(max(max(max(data))))
disp(min(min(min(data))))

phi_a_2d = reshape(data,[nx(2), nx(1)]);
image(phi_a_2d(:,:),'CDataMapping','scaled');
set(gca,'DataAspectRatio',[1 1 1]); % need to be changed...
axis off;
c = jet(1024);
colormap(c)
caxis([0.0 1])
colorbar('FontSize',20)
%caxis([-0.45 0.45])

set(h, 'PaperPositionMode', 'auto');
% [ auto | {manual} ]
set(h, 'PaperUnits', 'points');
% [ {inches} | centimeters | normalized | points ]
set(h, 'PaperPosition', [0 0 500 500]);
% [left,bottom,width,height]
%[~,outfilename,~] = fileparts(filename);
print (h,"2d_density",'-dpng') % print (h,'bulk','-dpdf')
%print (h,strcat("pressure_", outfilename),'-dpng') % print (h,'bulk','-dpdf')
%close(h)