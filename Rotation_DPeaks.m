function cl = Rotation_DPeaks(points_c,dc,data_sparse)

K = 6;
[idx_cell,d_cell] = rangesearch(points_c,points_c,dc);

ND=length(points_c(:,1));
for i=1:ND
  rho(i)=0.;
  delta(i)=0.;
  nneigh(i)=0.;
end
for i=1:ND
    rho(i) = length(cell2mat(idx_cell(i)))-1;
end
[rho_sorted,ordrho]=sort(rho,'descend');
count_density_peak = 0;
count_outliers = 0;
flag_density_peak = 0;
local_density_peak_first = [];
%
for i=1:ND
    if rho(i) == 0
        count_outliers = count_outliers + 1;
        outliers(count_outliers) = i;
        continue;
    end
    temp_d = cell2mat(d_cell(i))';
    temp_id = cell2mat(idx_cell(i))';
    temp_rho = zeros(length(temp_id),1);  
    for j=1:length(temp_id)
        temp_rho(j) = rho(temp_id(j));
    end
    temp_id_d_rho = [temp_id temp_d temp_rho];
    temp_id_d_rho_sortrho = sortrows(temp_id_d_rho,-3);
    if temp_id_d_rho_sortrho(1,1) == i  || temp_id_d_rho_sortrho(1,3) == rho(i)
        count_density_peak = count_density_peak + 1;
        local_density_peak(count_density_peak) = i;
        local_density_peak_first = [local_density_peak_first;points_c(i,:)];
        continue;
    end
    for j=1:length(temp_id)
        if(i == temp_id_d_rho_sortrho(j,1))
            temp_marix = temp_id_d_rho_sortrho(1:j-1,:);
            temp_marix_sortdelta = sortrows(temp_marix,-2);
            flag = 0;
            for w= j:length(temp_id_d_rho_sortrho(:,1))
                if temp_id_d_rho_sortrho(w,2) ~= 0
                   if temp_id_d_rho_sortrho(w,1)<i && temp_id_d_rho_sortrho(w,3) == temp_id_d_rho_sortrho(j,3)...
                                    && temp_id_d_rho_sortrho(j-1,2) ~= 0
                      delta(i) = temp_id_d_rho_sortrho(w,2);
                      nneigh(i) = temp_id_d_rho_sortrho(w,1);
                      flag = 1;
                    end
                    break;
                end
            end
            if flag == 1
               break;
            end
            length_marix = length(temp_marix_sortdelta(:,1));
            for o= length_marix:-1:1
               if temp_marix_sortdelta(1,2) == temp_marix_sortdelta(length_marix,2)
                  delta(i) = temp_marix_sortdelta(1,2);
                  nneigh(i) = temp_marix_sortdelta(1,1);
               end
               if temp_marix_sortdelta(o,2) == temp_marix_sortdelta(length_marix,2)
                  continue;
               end
               delta(i) = temp_marix_sortdelta(o+1,2);
               nneigh(i) = temp_marix_sortdelta(o+1,1);
               break;
            end
            break;
        end

    end
    
end

dc_plus = dc;
count_density_peak_plus = 0;
local_density_peak_plus = [];
if count_density_peak > 1
    for i=2:K
        dc_plus = dc_plus*i;
        i
        length(local_density_peak)
%         [idx_cell_plus,d_cell_plus] = rangesearch(points_c,points_c,dc_plus);    
        for j=1:length(local_density_peak)
            local_density_peak(j)
            [idx_cell_plus,d_cell_plus] = rangesearch(points_c(local_density_peak(j),:),points_c,dc_plus);
            temp_d = [];
            temp_id = [];
            for c=1:ND
                if cell2mat(idx_cell_plus(c)) == 1
                    temp_id = [temp_id ; c];
                    temp_d = [temp_d ; cell2mat(d_cell_plus(c))];
                end
                
            end
%             temp_d = cell2mat(d_cell_plus(1))';
%             temp_id = cell2mat(idx_cell_plus(1))';
            temp_rho = zeros(length(temp_id),1);
            for k=1:length(temp_id)
                temp_rho(k) = rho(temp_id(k));
            end
            temp_id_d_rho = [temp_id temp_d temp_rho];
            temp_id_d_rho_sortrho = sortrows(temp_id_d_rho,-3);
            if temp_id_d_rho_sortrho(1,1) == local_density_peak(j) || temp_id_d_rho_sortrho(1,3) == rho(local_density_peak(j))
                count_density_peak_plus = count_density_peak_plus + 1;
                local_density_peak_plus(count_density_peak_plus) = local_density_peak(j);
                continue;
            end
            for k=1:length(temp_id)
                if(local_density_peak(j) == temp_id_d_rho_sortrho(k,1))
                    temp_marix = temp_id_d_rho_sortrho(1:k-1,:);
                    temp_marix_sortdelta = sortrows(temp_marix,-2);
                    flag = 0;
                    for w= k:length(temp_id_d_rho_sortrho(:,1))
                        if temp_id_d_rho_sortrho(w,2) ~= 0
                            if temp_id_d_rho_sortrho(w,1)<local_density_peak(j) && temp_id_d_rho_sortrho(w,3) == temp_id_d_rho_sortrho(k,3)...
                                    && temp_id_d_rho_sortrho(k-1,2) ~= 0
                                delta(local_density_peak(j)) = temp_id_d_rho_sortrho(w,2);
                                nneigh(local_density_peak(j)) = temp_id_d_rho_sortrho(w,1);
                                flag = 1;
                            end
                            break;
                        end
                    end
                    if flag == 1
                       break;
                    end
                    length_marix = length(temp_marix_sortdelta(:,1));
                    for o= length_marix:-1:1
                        if temp_marix_sortdelta(1,2) == temp_marix_sortdelta(length_marix,2)
                            delta(local_density_peak(j)) = temp_marix_sortdelta(1,2);
                            nneigh(local_density_peak(j)) = temp_marix_sortdelta(1,1);
                        end
                        if temp_marix_sortdelta(o,2) == temp_marix_sortdelta(length_marix,2)
                            continue;
                        end
                        delta(local_density_peak(j)) = temp_marix_sortdelta(o+1,2);
                        nneigh(local_density_peak(j)) = temp_marix_sortdelta(o+1,1);
                        break;
                    end
                    break;
                end              
            end
        end
       if length(local_density_peak_plus) == 1
            break;
       end
       
       count_local_density_peak_plus = 1;
       flag = 0;
       for u = 2:length(local_density_peak_plus)
           if rho(local_density_peak_plus(1)) == rho(local_density_peak_plus(u))
               count_local_density_peak_plus = count_local_density_peak_plus + 1;
           else
               break;
           end
           if count_local_density_peak_plus == length(local_density_peak_plus)
               flag = 1;
           end
       end
       if flag == 1
           break;
       end
       if length(local_density_peak) == length(local_density_peak_plus)
           break;
       end
       local_density_peak = [];
       local_density_peak = local_density_peak_plus;
       local_density_peak_plus = [];
       count_density_peak_plus = 0;
       
       
    end
end
length(local_density_peak_plus)
if length(local_density_peak_plus) > 1
    fprintf('注意local_density_peak_plus的长度大于1，其值为%d\n',length(local_density_peak_plus));
    delta(local_density_peak_plus(1)) = max(delta(:));
    for i=2:length(local_density_peak_plus)
%         density_peak_pdist = pdist2(local_density_peak_plus(),points_c);
        
        
        delta(local_density_peak_plus(i)) = pdist2(points_c(local_density_peak_plus(i)),points_c(local_density_peak_plus(1)));
        nneigh(local_density_peak_plus(i)) = local_density_peak_plus(1);

%         temp_d = cell2mat(d_cell_plus(local_density_peak_plus(i)))';
%         delta(local_density_peak_plus(i)) = temp_d(local_density_peak_plus(1));
%         nneigh(local_density_peak_plus(i)) = local_density_peak_plus(1);
    end
end
if length(local_density_peak_plus) == 1
    delta(local_density_peak_plus(1)) = max(delta(:));
end

dc_plus = dc;
count_outliers_plus = 0;
outliers_plus = [];
if count_outliers > 0
    for i=2:ND
        dc_plus = dc_plus*i;
%         [idx_cell_plus,d_cell_plus] = rangesearch(points_c,points_c,dc_plus);
        
        for j=1:length(outliers)
            [idx_cell_plus,d_cell_plus] = rangesearch(points_c(outliers(j),:),points_c,dc_plus);
            temp_d = [];
            temp_id = [];
            for c=1:ND
                if cell2mat(idx_cell_plus(c)) == 1
                    temp_id = [temp_id ; c];
                    temp_d = [temp_d ; cell2mat(d_cell_plus(c))];
                end
                
            end
            temp_rho = zeros(length(temp_id),1);
            for k=1:length(temp_id)
                temp_rho(k) = rho(temp_id(k));
            end
            temp_id_d_rho = [temp_id temp_d temp_rho];
            temp_id_d_rho_sortrho = sortrows(temp_id_d_rho,-3);
            if temp_id_d_rho_sortrho(1,1) == outliers(j) || temp_id_d_rho_sortrho(1,3) == rho(outliers(j))
                count_outliers_plus = count_outliers_plus + 1;
                outliers_plus(count_outliers_plus) = outliers(j);
                continue;
            end
            
            for k=1:length(temp_id)
                if(outliers(j) == temp_id_d_rho_sortrho(k,1))
                    temp_marix = temp_id_d_rho_sortrho(1:k-1,:);
                    temp_marix_sortdelta = sortrows(temp_marix,-2);
                    flag = 0;
                    for w= k:length(temp_id_d_rho_sortrho(:,1))
                        if temp_id_d_rho_sortrho(w,2) ~= 0
                            if temp_id_d_rho_sortrho(w,1)<outliers(j) && temp_id_d_rho_sortrho(w,3) == temp_id_d_rho_sortrho(k,3)...
                                    && temp_id_d_rho_sortrho(k-1,2) ~= 0
                                delta(outliers(j)) = temp_id_d_rho_sortrho(w,2);
                                nneigh(outliers(j)) = temp_id_d_rho_sortrho(w,1);
                                flag = 1;
                            end
                            break;
                        end
                    end
                    if flag == 1
                       break;
                    end
                    length_marix = length(temp_marix_sortdelta(:,1));
                    for o= length_marix:-1:1
                        if temp_marix_sortdelta(1,2) == temp_marix_sortdelta(length_marix,2)
                            delta(outliers(j)) = temp_marix_sortdelta(1,2);
                            nneigh(outliers(j)) = temp_marix_sortdelta(1,1);
                        end
                        if temp_marix_sortdelta(o,2) == temp_marix_sortdelta(length_marix,2)
                            continue;
                        end
                        delta(outliers(j)) = temp_marix_sortdelta(o+1,2);
                        nneigh(outliers(j)) = temp_marix_sortdelta(o+1,1);
                        break;
                    end
                    break;
                end              
            end
            
        end
        
        
        if length(outliers) == 0
            break;
        end
        outliers = [];
        outliers = outliers_plus;
        outliers_plus = [];
        count_outliers_plus = 0;
        
    end
end


[rho,PS] = mapminmax(rho,0,1);
[delta,PS] = mapminmax(delta,0,1);
[delta_sorted,orddelta]=sort(delta,'descend');
[rho_sorted,ordrho]=sort(rho,'descend');

% disp('Generated file:DECISION GRAPH')
% disp('column 1:Density')
% disp('column 2:Delta')
% 
% fid = fopen('DECISION_GRAPH', 'w');
% for i=1:ND
%    fprintf(fid, '%6.2f %6.2f\n', rho(i),delta(i));
% end

% disp('Select a rectangle enclosing cluster centers')
% scrsz = get(0,'ScreenSize');
% figure('Position',[6 72 scrsz(3)/4. scrsz(4)/1.3]);
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
if data_sparse 
a=1;%parameter"a"2  a越小越宽,defult=1
c=0.6;%perameter"c"0.45
m=-0 ;%perameter"m"
theta=pi/(7);%perameter"m"越小越往右
else
a=8;%parameter"a"2  a越小越宽
c=0.15;%perameter"c"0.45
m=-0 ;%perameter"m"
theta=pi/(6);%perameter"m"越小越往右
end

ytest=a*xtest.^2+ c;
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

T=[cos(theta) -sin(theta);
   sin(theta) cos(theta)];

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
y=a*x.^2+ c;
x2=-1:0.001:1;
y2=2*x2.^2+ 0.33;
plot(x,y,'Color',[0.945 0.463 0.4]);
q=0;
hold on
plot(0,x,'Color','r');
% plot(x2,y2,'--','Color',[0.470588235294118 0.670588235294118 0.188235294117647]);
xlabel ('X')
ylabel ('Y')
z=a*new_data(:,1).^2+c;
ids=find(new_data(:,2)>z);
f1 = @(x_e) x_e*cosd(90);
f2 = @(x_e) a*x_e.^2+c;
f3 = @(x_e) x_e*cosd(200);
f4 = @(x_e) x_e*cosd(45)+ 3;

x1 = linspace(-1, m, 500);
plot(x1, f1(x1), 'Visible','off')
plot(x1, f2(x1), 'Visible','off')

x2 = linspace(-1, m, 500);
plot(x2, f3(x2), 'Visible','off')
plot(x2, f4(x2), 'Visible','off')

x_range = x2((f4(x2)>f3(x2)) & (x2<=0));  % 确定待填充区域的x范围
[y1, y2, y3, y4] = deal(f1(x_range), f2(x_range), f3(x_range), f4(x_range));
y_upper = min(y2, y4);  % 填充区域的上边界
y_lower = max(y1, y3);  % 填充区域的下边界

figure(1);
patch([x_range, fliplr(x_range)], [y_lower, fliplr(y_upper)], 'g','LineStyle','none')
alpha(0.5)
% x_t=[m,-1,-1,m];y_t=[0,0,0.94,0];
x_t=[0,-1,-1,0];y_t=[0,0,0.94,0];
fill(x_t,y_t,'g','LineStyle','none');
alpha(0.5)
line([m,m],[0,1.6],'Color','b');
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