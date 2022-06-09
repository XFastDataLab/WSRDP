function [cl, ids] = WSRDP_Rotation_DPeaks(points_c,dc,wsrdp_a,wsrdp_c,wsrdp_m,wsrdp_theta)

[rho, delta, ordrho, nneigh] = WSRDP_FastDistence(points_c, dc);
ND=length(points_c(:,1));

for i=1:ND
  ind(i)=i;
  gamma(i)=rho(i)*delta(i);
end
% subplot(2,1,1)
% figure(1)
% tt=plot(rho(:),delta(:),'*','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
xtest=-1:.1:1;
%新s3参数
%y=1.7*x.^2+ 0.27;data set6,persent4,a4,c0.22,theta=pi/6
%ym散点图参数a=2 c=0.45

ytest=wsrdp_a*xtest.^2+ wsrdp_c;
X=xtest*cosd(-45)-ytest*sind(-45);
Y=xtest*sind(-45)+ytest*cosd(-45);

% hold on
% % plot(X,Y);
% axis([0 1 0 1]);
% title ('Decision Graph','FontSize',15.0)
% xlabel ('\rho')
% ylabel ('\delta')

%fig=subplot(2,1,1)
% rect = getrect(fig);
% rhomin=rect(1);
% deltamin=rect(2);
NCLUST=0;    
for i=1:ND
  cl(i)=-1; 
end

T=[cos(wsrdp_theta) -sin(wsrdp_theta);
   sin(wsrdp_theta) cos(wsrdp_theta)];

deta_gamma_delta=[rho; delta];

new_data=T*deta_gamma_delta;
new_data=new_data';
figure(1);
%subplot(2,1,2)
plot(new_data(:,1),new_data(:,2),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
new_rho=new_data(:,1);
new_delta=new_data(:,2);
axis([-0.9 0.9 0 1.6]);
hold on 
x=-1:0.001:1;
y=wsrdp_a*x.^2+ wsrdp_c;
x2=-1:0.001:1;
y2=2*x2.^2+ 0.33;
plot(x,y,'Color',[0.945 0.463 0.4]);
q=0;
hold on
plot(0,x,'Color','r');
% plot(x2,y2,'--','Color',[0.470588235294118 0.670588235294118 0.188235294117647]);
xlabel ('X')
ylabel ('Y')
z=wsrdp_a*new_data(:,1).^2+wsrdp_c;
ids=find(new_data(:,2)>z);
f1 = @(x_e) x_e*cosd(90);
f2 = @(x_e) wsrdp_a*x_e.^2+wsrdp_c;
f3 = @(x_e) x_e*cosd(200);
f4 = @(x_e) x_e*cosd(45)+ 3;

x1 = linspace(-1, wsrdp_m, 500);
plot(x1, f1(x1), 'Visible','off')
plot(x1, f2(x1), 'Visible','off')

x2 = linspace(-1, wsrdp_m, 500);
plot(x2, f3(x2), 'Visible','off')
plot(x2, f4(x2), 'Visible','off')

x_range = x2((f4(x2)>f3(x2)) & (x2<=0));  % 确定待填充区域的x范围
[y1, y2, y3, y4] = deal(f1(x_range), f2(x_range), f3(x_range), f4(x_range));
y_upper = min(y2, y4);  % 填充区域的上边界
y_lower = max(y1, y3);  % 填充区域的下边界

figure(1);
patch([x_range, fliplr(x_range)], [y_lower, fliplr(y_upper)], 'g','LineStyle','none')
alpha(0.5)
% x_t=[wsrdp_m,-1,-1,wsrdp_m];y_t=[0,0,0.94,0];
x_t=[0,-1,-1,0];y_t=[0,0,0.94,0];
fill(x_t,y_t,'g','LineStyle','none');
alpha(0.5)
line([wsrdp_m,wsrdp_m],[0,1.6],'Color','b');
for i=1:length(ids)
    NCLUST=NCLUST+1;
    cl(ids(i))=NCLUST;  
    icl(NCLUST)=ids(i);
end
%assignation
for i=1:ND
  if (cl(ordrho(i))==-1)
    cl(ordrho(i))=cl(nneigh(ordrho(i)));
  end
end
cmap=colormap;
for i=1:NCLUST
   ic=int8((i*60.)/(NCLUST*1.));
   figure(1)
   hold on
   plot(new_rho(icl(i)),new_delta(icl(i)),'o','MarkerSize',7,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
end