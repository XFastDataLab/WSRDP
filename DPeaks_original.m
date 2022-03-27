function cl = DPeaks_original(points,gamma_num,dc)

K = 5;
[idx_cell,d_cell] = rangesearch(points,points,dc);

ND=length(points(:,1));
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
%         [idx_cell_plus,d_cell_plus] = rangesearch(points,points,dc_plus);    
        for j=1:length(local_density_peak)
            local_density_peak(j)
            [idx_cell_plus,d_cell_plus] = rangesearch(points(local_density_peak(j),:),points,dc_plus);
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
%         density_peak_pdist = pdist2(local_density_peak_plus(),points);
        
        
        delta(local_density_peak_plus(i)) = pdist2(points(local_density_peak_plus(i)),points(local_density_peak_plus(1)));
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
%         [idx_cell_plus,d_cell_plus] = rangesearch(points,points,dc_plus);
        
        for j=1:length(outliers)
            [idx_cell_plus,d_cell_plus] = rangesearch(points(outliers(j),:),points,dc_plus);
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


% ND=length(points(:,1));
% NL=max(xx(:,1));
% if (NL>ND)
%   ND=NL;
% end
% N=size(xx,1);
% for i=1:ND
%   for j=1:ND
%     dist(i,j)=0;
%   end
% end
% for i=1:N
%   ii=xx(i,1);
%   jj=xx(i,2);
%   dist(ii,jj)=xx(i,3);
%   dist(jj,ii)=xx(i,3);
% end
% percent=2;
% fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);
% 
% position=round(N*percent/100);
% sda=sort(xx(:,3));
% dc=sda(position);
% 
% fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);

% 
% for i=1:ND
%   rho(i)=0.;
% end
% %
% % Gaussian kernel
% %
% for i=1:ND-1
%   for j=i+1:ND
%      rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
%      rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
%   end
% end
% %
% % "Cut off" kernel
% %
% % for i=1:ND-1
% %  for j=i+1:ND
% %    if (dist(i,j)<dc)
% %       rho(i)=rho(i)+1.;
% %       rho(j)=rho(j)+1.;
% %    end
% %  end
% % end
% 
% maxd=max(max(dist));
% 
% [rho_sorted,ordrho]=sort(rho,'descend');
% delta(ordrho(1))=-1.;
% nneigh(ordrho(1))=0;
% 
% for ii=2:ND
%    delta(ordrho(ii))=maxd;
%    for jj=1:ii-1
%      if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
%         delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
%         nneigh(ordrho(ii))=ordrho(jj);
%      end
%    end
% end
% delta(ordrho(1))=max(delta(:));
% disp('Generated file:DECISION GRAPH')
% disp('column 1:Density')
% disp('column 2:Delta')






fid = fopen('DECISION_GRAPH', 'w');
for i=1:ND
   fprintf(fid, '%6.2f %6.2f\n', rho(i),delta(i));
end

disp('Select a rectangle enclosing cluster centers')
scrsz = get(0,'ScreenSize');
figure('Position',[6 72 scrsz(3)/4. scrsz(4)/1.3]);
for i=1:ND
  ind(i)=i;
  gamma(i)=rho(i)*delta(i);
end
gamma=gamma';
rho=rho';
delta=delta';

NDIndex = linspace(1,ND,ND);
NDIndex = NDIndex';
gamma=[gamma rho delta NDIndex];
%subplot(2,1,1);
figure(1)
plot(rho(:),delta(:),'o','MarkerSize',10,'MarkerFaceColor','k','MarkerEdgeColor','k');

gamma_sorted = sortrows(gamma,1);

gamma_rho=sortrows(gamma_sorted(ND-gamma_num+1:ND,:),2)
gamma_delta=sortrows(gamma_sorted(ND-gamma_num+1:ND,:),3)

% NCLUST=gamma_num;

xtest2=0:.5:160;
ytest2=gamma_sorted(ND-gamma_num+1,1)./xtest2.^1;
hold on
figure(1)
 plot(xtest2,ytest2,'b');
axis([0 120 0 0.4]);
axis off



%示例决策图的画画代码
% annotation('textbox',[0.293332268370606 0.466926070038911 0.02455910543131 0.0311284046692609],'Color',[0 0 1],'String','t',...
%     'FontWeight','bold','FontSize',14,'FitBoxToText','off', 'EdgeColor',[1 1 1],'FontAngle','italic');
% annotation('textbox',[0.786942492012777 0.301556420233463 0.0245591054313099 0.0311284046692609],'Color',[0 0 1],'String','p',...
%     'FontWeight','bold','FontSize',14,'FitBoxToText','off','EdgeColor',[1 1 1],'FontAngle','italic');
% annotation('textbox',[0.850840255591053 0.79182879377432 0.0245591054313099 0.0311284046692609],'Color',[0 0 1],'String','q',...
%     'FontWeight','bold','FontSize',14,'FitBoxToText','off','EdgeColor',[1 1 1],'FontAngle','italic');
% annotation('textarrow',[0.640660437267528 0.582713417399979],[0.361542525515307 0.334025415249148],'String',{'\rho \times \delta = 235.62'},...
%     'Color',[0.945 0.463 0.4],'LineWidth',1,'FontWeight','bold','FontSize',12,'FontAngle','italic');
% annotation('line',[0.393388429752066 0.391239966170601],[0.261758691206544 0.923933209647495],'Color',[0.470588235294118 0.670588235294118 0.188235294117647],...
%     'LineWidth',2);
% annotation('line',[0.393388429752066 0.905821917808219],[0.263803680981595 0.257254337141555],'Color',[0.470588235294118 0.670588235294118 0.188235294117647],...
%     'LineWidth',2);
% annotation('line',[0.904132231404959 0.905785123966942],[0.924335378323109 0.255623721881391],'Color',[0.470588235294118 0.670588235294118 0.188235294117647],...
%     'LineWidth',2);
% annotation('line',[0.388429752066116 0.902479338842975],[0.920245398773006 0.920245398773006],'Color',[0.470588235294118 0.670588235294118 0.188235294117647],...
%     'LineWidth',2);
% % title ('Original Decision Graph','FontSize',15.0)
% matrix_s=matrix_s';
% for i=1:size(matrix_s)
%     hold on
%     plot(rho(matrix_s(i)),delta(matrix_s(i)),'o','MarkerSize',6,'MarkerFaceColor','r','MarkerEdgeColor','r')
% end
% xtest3=-100:1:100;
% ytest3=0.08*xtest3.^2+ sqrt(12.61^2+12.22^2);
% X=xtest3*cosd(-45)-ytest3*sind(-45);
% Y=xtest3*sind(-45)+ytest3*cosd(-45);
% hold on
% plot(X,Y);
% xlabel ('\rho')
% ylabel ('\delta')






%yanmns
rhomin=gamma_rho(1,2)
deltamin=gamma_delta(1,3)
rho=rho';
delta=delta';

for i=1:ND
  cl(i)=-1;
end

% 旋转45度决策图加画画
% theta=pi/4;
% T=[cos(theta) -sin(theta);
%    sin(theta) cos(theta)];
% deta_gamma_delta=[rho; delta];
% new_data=T*deta_gamma_delta;
% new_data=new_data';
% figure(2);
% %subplot(2,1,2)
% plot(new_data(:,1),new_data(:,2),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
% new_rho=new_data(:,1);
% new_delta=new_data(:,2);
% hold on 
% x=-50:1:50;
% a=0.05;
% c=17.56;
% y=a*x.^2+ c;
% plot(x,y,'Color',[0.101 0.58 0.737]);
% x2=-50:1:50;
% y2=a*x.^2;
% plot(x2,y2,'Color','r','LineStyle','--');
% xlabel ('X')
% ylabel ('Y')
% axis([-30 40 0 52]);
% annotation('textbox',[0.430767365661861 0.41916971916972 0.0252123197903015 0.0341880341880342],'Color',[0 0 1],'String','t',...
%     'FontWeight','bold','LineStyle','none','FontSize',14,'FitBoxToText','off', 'FitBoxToText','off','FontAngle','italic');
% annotation('textbox',[0.720184141546527 0.699023199023199 0.0225910878112713 0.0307692307692308],'Color',[0 0 1],'String','p',...
%     'FontWeight','bold','LineStyle','none','FontSize',14,'FitBoxToText','off','FitBoxToText','off','FontAngle','italic');
% annotation('textbox',[0.614859764089122 0.908974358974361 0.018659239842726 0.0290598290598292],'Color',[0 0 1],'String','q',...
%    'FontWeight','bold','LineStyle','none','FontSize',14,'FitBoxToText','off','FitBoxToText','off','FontAngle','italic');
% annotation('textarrow',[0.335714285714286 0.367857142857143],[0.387095238095239 0.428571428571429],'String',{'y=0.05*x^2+17.56'},...
%      'Color',[0.101 0.58 0.737],'LineWidth',1,'FontWeight','bold','FontAngle','italic');
% annotation('textbox',[0.152785714285714 0.402380952380954 0.213285714285715 0.0261904761904788],'String',{'dicision curve'},...
%     'Color',[0.101 0.58 0.737],'LineStyle','none','FontWeight','bold','FontAngle','italic','FitBoxToText','off');
% annotation('textbox',[0.6885 0.211904761904762 0.172214285714286 0.0380952380952393],'Color',[1 0 0],'String',{'y=0.05*x^2'},...
%     'LineStyle','none','FontWeight','bold','FontAngle','italic','FitBoxToText','off');
% annotation('textarrow',[0.680357142857143 0.630357142857143],[0.249 0.276190476190476],'Color',[1 0 0],'String',{'base curve'},'LineWidth',1,...
%     'FontWeight','bold','FontAngle','italic');
% annotation('doublearrow',[0.464285714285714 0.467857142857143],[0.376190476190476 0.111904761904762],'Color',[0.470588235294118 0.670588235294118 0.188235294117647],...
%     'LineWidth',1,'LineStyle','--');
% annotation('textarrow',[0.416071428571429 0.455357142857143],[0.293874899112187 0.285351089588378],'Color',[0.470588235294118 0.670588235294118 0.188235294117647],...
%     'String',{'decison gap of t '},'FontSize',12,'FontWeight','bold','FontAngle','italic');
% annotation('doublearrow',[0.658928571428571 0.653571428571429],[0.90952380952381 0.352380952380952],'Color',[0.470588235294118 0.670588235294118 0.188235294117647],'LineWidth',1,'LineStyle','--');
% annotation('textarrow',[0.607142857142856 0.64642857142857],[0.772446327683619 0.76392251815981],'Color',[0.470588235294118 0.670588235294118 0.188235294117647],...
%     'String',{'decison gap of q'},'FontWeight','bold','FontSize',12,'FontAngle','italic');
% matrix_s=matrix_s';
% for i=1:size(matrix_s)
%     hold on
%     plot(new_rho(matrix_s(i)),new_delta(matrix_s(i)),'o','MarkerSize',6,'MarkerFaceColor','r','MarkerEdgeColor','r')
% end
% title ('New Decision Graph','FontSize',15.0)

%原算法算中心点
% NCLUST = 0;
% for i=1:ND
%   if ( (rho(i)>=rhomin) && (delta(i)>=deltamin))
%      NCLUST=NCLUST+1;
%      cl(i)=NCLUST;
%      icl(NCLUST)=i;
%   end
% end

NCLUST=gamma_num;
icl = gamma_rho(:,4);
for i=1:NCLUST
    cl(icl(i)) = i;
end

fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);
disp('Performing assignation')

%assignation
for i=1:ND
  if (cl(ordrho(i))==-1)
    cl(ordrho(i))=cl(nneigh(ordrho(i)));
  end
end
%halo
for i=1:ND
  halo(i)=cl(i);
end
bord_region=zeros(2,ND);
for i=1:ND
    bord_region(:,i)=cl(i);
end
if (NCLUST>1)
  for i=1:NCLUST
    bord_rho(i)=0.;
  end
  for i=1:NCLUST
    bord_rho_num(i)=0.;
  end
  bord_region_num = 0;
  for i=1:ND-1
    for j=(i+1):ND
      if ((cl(i)~=cl(j))&& (dist(i,j)<=dc))
%         rho_aver=(rho(i)+rho(j))/2.;
        bord_region(2,i)=0;
        bord_region_num = bord_region_num+1;
        if (rho(i)>bord_rho(cl(i)))
          fprintf('i=%d,j=%d,cl(i)=%d,bord_rho(cl(i))=%f  \n',i,j,cl(i),bord_rho(cl(i)));
          bord_rho(cl(i))=rho(i);
          bord_rho_num(cl(i))=i;
        end
        if (rho(j)>bord_rho(cl(j)))
          fprintf('i=%d,j=%d,cl(i)=%d,bord_rho(cl(i))=%f  \n',i,j,cl(i),bord_rho(cl(i)));
          bord_rho(cl(j))=rho(j);
          bord_rho_num(cl(j))=j;
        end
      end
    end
  end
  for i=1:ND
    if (rho(i)<bord_rho(cl(i)))
      halo(i)=0;
    end
  end
end
for i=1:NCLUST
  nc=0;
  nh=0;
  for j=1:ND
    if (cl(j)==i) 
      nc=nc+1;
    end
    if (halo(j)==i) 
      nh=nh+1;
    end
  end
  fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i \n', i,icl(i),nc,nh,nc-nh);
end

cmap=colormap;
hold on
for i=1:NCLUST
   ic=int8((i*64.)/(NCLUST*1.));
%    subplot(2,1,1)
%    figure(1)
   hold on
   plot(rho(icl(i)),delta(icl(i)),'o','MarkerSize',8,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
%    figure(2)
%    hold on
%    plot(new_rho(icl(i)),new_delta(icl(i)),'o','MarkerSize',8,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
end

% cl = xlsread('D:\PostgraduateStudy\Rotation-DPeak\Code\RotationDP_data\Record\cl_data3_Original.xlsx');
%subplot(2,1,2)
figure(3)
axis off
disp('Performing 2D nonclassical multidimensional scaling')
%title ('2D Nonclassical multidimensional scaling','FontSize',15.0)
xlabel ('X')
ylabel ('Y')
for i=1:ND
 A(i,1)=0.;
 A(i,2)=0.;
end
shapes='o*^+sx*<pd.^ph>dv';
for i=1:NCLUST
  nn=0;
  ic=int8((i*64.)/(NCLUST*1.));
  for j=1:ND
    if (cl(j)==i)
%     if (halo(j)==i)
      nn=nn+1;
      A(nn,1)=points(j,1);
      A(nn,2)=points(j,2);
    end
  end
  shapeindex= mod(icl(i),length(shapes))+1;
  hold on
%   if i==5
%   m4 = plot(A(1:nn,1),A(1:nn,2),shapes(1),'MarkerSize',10,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
%   continue;
%   end
%   if i==3
%   m5 = plot(A(1:nn,1),A(1:nn,2),shapes(9),'MarkerSize',10,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
%   continue;
%   end
  plot(A(1:nn,1),A(1:nn,2),shapes(shapeindex),'MarkerSize',6,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
end

%寻找border region算法
% for i=1:bord_region_num
%  B(i,1)=0.;
%  B(i,2)=0.;
% end
% for i=1:NCLUST
%   nn=0;
%   ic=int8((i*25.)/(NCLUST*1.));
%   for j=1:ND
%     if (bord_region(1,j)==i)
%       if (bord_region(2,j)==0)
%         nn=nn+1;
%         B(nn,1)=points(j,1);
%         B(nn,2)=points(j,2);
%       end
%     end
%   end
%   shapeindex= mod(i,length(shapes));
%   hold on
%   if i==3
%   m2 = plot(B(1:nn,1),B(1:nn,2),shapes(5),'MarkerSize',10,'MarkerFaceColor','k','MarkerEdgeColor','k');
%   continue;
%   end
%   if i==5
%   m3 = plot(B(1:nn,1),B(1:nn,2),shapes(5),'MarkerSize',10,'MarkerFaceColor','b','MarkerEdgeColor','b');
%   continue;
%   end
% %   plot(B(1:nn,1),B(1:nn,2),shapes(5),'MarkerSize',10,'MarkerFaceColor','k','MarkerEdgeColor','k');
% end


faa = fopen('CLUSTER_ASSIGNATION', 'w');
disp('Generated file:CLUSTER_ASSIGNATION')
disp('column 1:element id')
disp('column 2:cluster assignation without halo control')
disp('column 3:cluster assignation with halo control')
for i=1:ND
   fprintf(faa, '%i %i %i\n',i,cl(i),halo(i));
%    if halo(i)==0
%       h=plot(points(i,1),points(i,2),'.','MarkerSize',6,'MarkerEdgeColor','k');
%    end
end

for i=1:ND  %寻找局部密度峰值点
   for j=1:NCLUST
      if i==icl(j)
          m=plot(points(i,1),points(i,2),'o','MarkerSize',5,'linewidth',2,'MarkerEdgeColor','r');
      end
%       if i==bord_rho_num(j)
%           m1=plot(points(i,1),points(i,2),'o','MarkerSize',5,'linewidth',2,'MarkerEdgeColor','g');
%       end
   end
end

% communities_Matrix = zeros(NCLUST,ND)*nan;
% communities_count = 1;
% for j=1:NCLUST
%     for i=1:ND
%         if cl(i)==j
%             communities_Matrix(j,communities_count)=i;
%             communities_count = communities_count+1;
%         end
%     end
%     communities_count = 1;
% end

% for i=1:ND  %右边两簇
%    for j=1:NCLUST
%       if (j==5 || j==3)
%       if i==icl(j)
%           m=plot(points(i,1),points(i,2),'o','MarkerSize',7,'linewidth',3,'MarkerEdgeColor','r');
%       end
%       if i==bord_rho_num(j)
%           m1=plot(points(i,1),points(i,2),'o','MarkerSize',7,'linewidth',3,'MarkerEdgeColor','g');
%       end
%       end
%    end
% end

if ~exist('h','var')
    legend(m,'density peaks','Location','northeast');
%   legend([m,m2],'density peaks','border region','Location','northwest');
%   legend([m,m1,m4,m5,m2,m3],'density peaks','border point of highest density','cluster 1','cluster 2','border of cluster 1','border of cluster 2','Location','northwest');
else
%   legend([m,h,m1],'density peaks','Outliers','border point of highest density','Location','northwest');
  legend([m,h],'density peaks','Outliers','Location','northwest');
end

