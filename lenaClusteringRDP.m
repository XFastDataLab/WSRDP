clear;
tic
I = imread('D:\PostgraduateStudy\test1\Test_2.png'); 
figure, imshow(I);    % 读入分割目标
data_sparse = 1;
originalRAW = 128;
originalCOL = 256;

% load('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\Endo_test\data_frame073\frame073_20percent_points_c.mat');

A = reshape(I(:, :, 1), originalRAW*originalCOL, 1);    % 将RGB分量各转为kmeans使用的数据格式n行，一样一样本
B = reshape(I(:, :, 2), originalRAW*originalCOL, 1);
C = reshape(I(:, :, 3), originalRAW*originalCOL, 1);
A_double = im2double(A); 
B_double = im2double(B);
C_double = im2double(C);

%计算HSV
I_h = rgb2hsv(I);
A_h = reshape(I_h(:, :, 1), originalRAW*originalCOL, 1); 
B_h = reshape(I_h(:, :, 2), originalRAW*originalCOL, 1); 
C_h = reshape(I_h(:, :, 3), originalRAW*originalCOL, 1); 

%计算灰度值
% row_gray = rgb2gray(I);
% row_gray_re = reshape(row_gray, originalRAW*originalCOL, 1);
% row_gray_re = im2double(row_gray_re);

%计算行列梯度
% [row_gradient , line_gradient] = gradient(double(I));
% 
% A_row_gradient = reshape(row_gradient(:, :, 1), originalRAW*originalCOL, 1);    % 行梯度
% B_row_gradient = reshape(row_gradient(:, :, 2), originalRAW*originalCOL, 1);
% C_row_gradient = reshape(row_gradient(:, :, 3), originalRAW*originalCOL, 1);
% A_line_gradient = reshape(line_gradient(:, :, 1), originalRAW*originalCOL, 1);    % 列梯度
% B_line_gradient = reshape(line_gradient(:, :, 2), originalRAW*originalCOL, 1);
% C_line_gradient = reshape(line_gradient(:, :, 3), originalRAW*originalCOL, 1);

%计算X,Y坐标
% X = [];
% Y = [];
% for i=1:originalCOL
%     for j=originalRAW:-1:1
%         X = [X ; i];
%         Y = [Y ; j];
%     end
% end
% X = reshape(X, originalRAW*originalCOL, 1);
% Y = reshape(Y, originalRAW*originalCOL, 1);

% points = [A_h B_h C_h A_double B_double C_double];
points = [A_double B_double C_double];
% points = [A B C 
% c2 = kmeans(double(dat), 15);    % 使用聚类算法分为2类
% r2 = reshape(c2, 365, 365);     % 反向转化为图片形式
% figure, imshow(label2rgb(r2))   % 显示分割结果


% newRAW = ceil(originalRAW*0.4);
% newCOL = ceil(originalCOL*0.4);
newRAW = ceil(originalRAW);
newCOL = ceil(originalCOL);

%图像压缩
% [x,y,z]=size(points);
% takePointsRow = linspace(1,originalRAW,newRAW);
% for i=1:newRAW
%     takePointsRow(i)=ceil(takePointsRow(i));
% end
% newPoints = [];
% count = 1;
% for i=1:x
%     if mod(i,originalRAW) == takePointsRow(count)
%         temp = points(i,:);
%         newPoints = [newPoints ; temp];
%         count = count + 1;
%     end
%     if count == newRAW
%         count = 1;
%     end
% end
% newX = (newRAW-1)*originalCOL;
% count = 1;
% newA = [];
% newB = [];
% newC = [];
% newD = [];
% newE = [];
% newF = [];
% 
% for i=1:newX
%      temp = mod(i,newRAW-1);
%      if temp == 0
%         temp = temp + newRAW-1; 
%      end 
%      newA(temp,count) = newPoints(i,1);
%      newB(temp,count) = newPoints(i,2);
%      newC(temp,count) = newPoints(i,3);
%      newD(temp,count) = newPoints(i,4);
%      newE(temp,count) = newPoints(i,5);
%      newF(temp,count) = newPoints(i,6);
%      if temp == newRAW-1
%         count = count + 1; 
%      end
% end
% 
% takePointsCol = linspace(1,originalCOL,newCOL);
% for i=1:newCOL
%     takePointsCol(i)=ceil(takePointsCol(i));
% end
% countCol = 1;
% newA_ex = [];
% newB_ex = [];
% newC_ex = [];
% newD_ex = [];
% newE_ex = [];
% newF_ex = [];
% for i=1:originalCOL
%     if i == takePointsCol(countCol)
%         countCol = countCol + 1;
%         newA_ex = [newA_ex , newA(:,i)];
%         newB_ex = [newB_ex , newB(:,i)];
%         newC_ex = [newC_ex , newC(:,i)];
%         newD_ex = [newD_ex , newD(:,i)];
%         newE_ex = [newE_ex , newE(:,i)];
%         newF_ex = [newF_ex , newF(:,i)];
%     end
% end
% 
% newI(:,:,1) = newA_ex;
% newI(:,:,2) = newB_ex;
% newI(:,:,3) = newC_ex;
% newII(:,:,1) = newD_ex;
% newII(:,:,2) = newE_ex;
% newII(:,:,3) = newF_ex;
% 
% A_C = reshape(newI(:, :, 1), (newRAW-1)*newCOL , 1);    
% B_C = reshape(newI(:, :, 2), (newRAW-1)*newCOL , 1);
% C_C = reshape(newI(:, :, 3), (newRAW-1)*newCOL , 1);
% D_C = reshape(newII(:, :, 1), (newRAW-1)*newCOL , 1);    
% E_C = reshape(newII(:, :, 2), (newRAW-1)*newCOL , 1);
% F_C = reshape(newII(:, :, 3), (newRAW-1)*newCOL , 1);
% 
% points_c = [A_C B_C C_C D_C E_C F_C];


[x,y,z]=size(points);
takePointsRow = linspace(1,originalRAW,newRAW);
for i=1:newRAW
    takePointsRow(i)=ceil(takePointsRow(i));
end
newPoints = [];
count = 1;
for i=1:x
    if mod(i,originalRAW) == takePointsRow(count)
        temp = points(i,:);
        newPoints = [newPoints ; temp];
        count = count + 1;
    end
    if count == newRAW
        count = 1;
    end
end
newX = (newRAW-1)*originalCOL;
count = 1;
newA = [];
newB = [];
newC = [];

for i=1:newX
     temp = mod(i,newRAW-1);
     if temp == 0
        temp = temp + newRAW-1; 
     end 
     newA(temp,count) = newPoints(i,1);
     newB(temp,count) = newPoints(i,2);
     newC(temp,count) = newPoints(i,3);
     if temp == newRAW-1
        count = count + 1; 
     end
end

takePointsCol = linspace(1,originalCOL,newCOL);
for i=1:newCOL
    takePointsCol(i)=ceil(takePointsCol(i));
end
countCol = 1;
newA_ex = [];
newB_ex = [];
newC_ex = [];

for i=1:originalCOL
    if i == takePointsCol(countCol)
        countCol = countCol + 1;
        newA_ex = [newA_ex , newA(:,i)];
        newB_ex = [newB_ex , newB(:,i)];
        newC_ex = [newC_ex , newC(:,i)];

    end
end

newI(:,:,1) = newA_ex;
newI(:,:,2) = newB_ex;
newI(:,:,3) = newC_ex;

A_C = reshape(newI(:, :, 1), (newRAW-1)*newCOL , 1);    
B_C = reshape(newI(:, :, 2), (newRAW-1)*newCOL , 1);
C_C = reshape(newI(:, :, 3), (newRAW-1)*newCOL , 1);

points_c = [A_C B_C C_C];

% 
% 
% %算dc
% [x,y]=size(points_c);
% %计算数量
% num= (x*x-x)/2;
% 
% distances=zeros(num,3);
% distMat= pdist2(points_c,points_c);
% k=1;
% loc=1;
% for i=1:x-1
%     distances(loc:loc+(x-k)-1,1)=k*ones(x-k,1);
%     distances(loc:loc+(x-k)-1,2)=(k+1:1:x);
%     distances(loc:loc+(x-k)-1,3)=distMat(k,k+1:x)';
%     loc=loc+(x-k);
%     k=k+1;
% end
% 
% %%%%%%RDP
% xx=distances;
% % points=pts3;
% 
% 
% ND=max(xx(:,2));
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
% percent=3;
% fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);
% 
% position=round(N*percent/100);
% sda=sort(xx(:,3));
% dc=sda(position);


% points_c = points;
% newRAW = originalRAW+1;
% newCOL = originalCOL;

K_num = 3;
c2 = kmeans(double(points_c), 5); 
r2 = reshape(c2, (newRAW-1), newCOL); 
% myColorMap = [0 255 0;255 0 0;0 0 255;255 255 0];%25500红  02550绿 00255蓝 2552550黄
% myColorMap = myColorMap/255;
figure, imshow(label2rgb(r2,colormap(lines)))  
% figure, imshow(label2rgb(r2))





dc = 0.02974;
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








% [x,y]=size(points_c);
% %计算数量
% num= (x*x-x)/2;
% 
% distances=zeros(num,3);
% distMat= pdist2(points_c,points_c);
% k=1;
% loc=1;
% for i=1:x-1
%     distances(loc:loc+(x-k)-1,1)=k*ones(x-k,1);
%     distances(loc:loc+(x-k)-1,2)=(k+1:1:x);
%     distances(loc:loc+(x-k)-1,3)=distMat(k,k+1:x)';
%     loc=loc+(x-k);
%     k=k+1;
% end
% 
% %%%%%%RDP
% xx=distances;
% % points=pts3;
% 
% 
% ND=max(xx(:,2));
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
% percent=3;
% fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);
% 
% position=round(N*percent/100);
% sda=sort(xx(:,3));
% dc=sda(position);
% 
% fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);
% 
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
% %for i=1:ND-1
% %  for j=i+1:ND
% %    if (dist(i,j)<dc)
% %       rho(i)=rho(i)+1.;
% %       rho(j)=rho(j)+1.;
% %    end
% %  end
% %end
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
a=1.002425;%parameter"a"2  a越小越宽
c=0.5;%perameter"c"0.45
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

%new halo
% for i=1:ND
%   halo(i)=cl(i);
% end
% if (NCLUST>1)
%     for i=1:ND
%         z=a*new_data(i,1).^2+c;
%        if ((new_data(i,1)<=m)&&(new_data(i,2)<z)) 
%         halo(i)=0;
%        end
%     end
%     for i=1:ND
%       halo_copy(i)=halo(i);
%     end
%     for i=1:ND
%       halo_e(i)=cl(i);
%     end
%     for e=1:4
%       for i=1:ND
%         if (halo(i)==0)
%               for j=1:ND
%                  if(nneigh(j)==i)
%                      if(halo_copy(j)==0)
%                         continue;
%                      else
%                        halo(j)=0;
%                        halo_e(j)=-2;
%                      end
%                  end
%               end
%         end
%       end
%     end
% end


% cl = cl';
r3 = reshape(cl, (newRAW-1), newCOL);     % 反向转化为图片形式
% length(r3(:,1))
% length(r3(1,:))
% count_one = 0;
% for i=1:length(r3(:,1))
%     for j=1:length(r3(1,:))
%         if r3(i,j) == -1
%            count_one = count_one + 1;
%         end
%     end
% end
% for k=1:length(r3(:,1))
%     for i=1:length(r3(:,1))
%         for j=1:length(r3(1,:))
%             if r3(i,j) == -1
%                 r3(i,j) = r3(i+1,j);
%                 if r3(i,j) ~= -1
%                     count_one = count_one - 1;
%                 end
%             end
%         end
%     end
%     if count_one == 0
%         break;
%     end
% end


% myColorMap = [92 34 35;0 255 0;255 255 255;255 0 0;255 255 0;0 0 255;238 44 121];%25500红  02550绿 00255蓝 2552550黄 92 34 35紫
% myColorMap = myColorMap/255;

% myColorMap = [255 255 255;128 128 128;0 0 0];
% myColorMap = myColorMap/255;
% figure, imshow(label2rgb(r3,colormap(myColorMap)));   % 显示分割结果
V = label2rgb(r3,colormap(lines));

%percent halo
% halo = cl;
% halo_plus = zeros(1,ND);
% percent_rho=40;
% percent_delta=40;
% des_percent_rho=rho_sorted(ceil(ND-0.01*percent_rho*ND));
% des_percent_delta=delta_sorted(ceil(0.01*percent_delta*ND));
% V_copy = V;
% outlisersRAW = newRAW-1;
% outlisersCOL = newCOL;
% % outlisersRAW = originalRAW;
% % outlisersCOL = originalCOL;
% for i=1:ND
%    if ((rho(i)<=des_percent_rho)&&(delta(i)>=des_percent_delta)) 
%      halo(i)=0;
%      V_col = ceil(i/outlisersRAW);
%      V_raw = mod(i,outlisersRAW);
%      if V_raw == 0 
%          V_raw = outlisersRAW;
%      end
%      V_raw
%      V_col
% 
%      [range_x_small , range_x_big , range_y_small , range_y_big] = deal(2);
%      if V_col - 1 == 0
%          range_y_small = 0;
%      end
%      if V_col - 2 == 0
%          range_y_small = 1;
%      end
%      if V_col + 0 == outlisersCOL
%          range_y_big = 0;
%      end
%      if V_col + 1 == outlisersCOL
%          range_y_big = 1;
%      end
%      if V_raw - 1 == 0
%          range_x_small = 0;
%      end
%      if V_raw - 2 == 0
%          range_x_small = 1;
%      end     
%      if V_raw + 0 == outlisersRAW
%          range_x_big = 0;
%      end     
%      if V_raw + 1 == outlisersRAW
%          range_x_big = 1;
%      end
% %      top_v_x = V_col - range_x_small;
% %      top_v_y = V_raw + range_y_big;
%      top_v_x = V_raw - range_x_small;
%      top_v_y = V_col + range_y_big;
%      range_marix = [];
%      for ii = 0:(range_x_small + range_x_big)
%         for jj = 0:(range_y_small + range_y_big)
%             temp_x = top_v_x + ii
%             temp_y = top_v_y - jj
%             temp_V = [V_copy(temp_x , temp_y , 1) V_copy(temp_x , temp_y , 2) V_copy(temp_x , temp_y , 3)];
%             range_marix = [range_marix ; temp_V];
%         end
%      end
%      data1=sortrows(range_marix);
%     [dataUnique,r]=unique(data1,'rows');
%     [dataUnique,r1]=unique(data1,'rows','last');
%     a_result = r1-r+1;
%     rang_marix_result = [dataUnique a_result];
%     rang_marix_result = sortrows(rang_marix_result,-4);
%      V(V_raw,V_col,1) = rang_marix_result(1,1);
%      V(V_raw,V_col,2) = rang_marix_result(1,2);
%      V(V_raw,V_col,3) = rang_marix_result(1,3);
%      
% %knnsearch处理噪声点
% %      [IDX , D] = knnsearch( local_density_peak_first, points_c(i,:) , 'k' ,1);
% %      V_col = ceil(i/outlisersRAW);
% %      V_raw = mod(i,outlisersRAW);
% %      if V_raw == 0 
% %          V_raw = outlisersRAW;
% %      end
% %      V_IDX_col = ceil(IDX/outlisersRAW);
% %      V_IDX_raw = mod(IDX,outlisersRAW);
% %      if V_IDX_raw == 0 
% %          V_IDX_raw = outlisersRAW;
% %      end
% % %      V(V_raw,V_col,1) = V(V_IDX_raw,V_IDX_col,1);
% % %      V(V_raw,V_col,2) = V(V_IDX_raw,V_IDX_col,2);
% % %      V(V_raw,V_col,3) = V(V_IDX_raw,V_IDX_col,3);
% %      V(V_raw,V_col,1) = 214;
% %      V(V_raw,V_col,2) = 123;
% %      V(V_raw,V_col,3) = 45;
%    end
% end
% halo_copy = halo;
% for e=1:4
%   for i=1:ND
%     if (halo(i)==0)
%           for j=1:ND
%              if(nneigh(j)==i)
%                  if(halo_copy(j)==0)
%                     continue;
%                  else
%                    halo(j)=0;
% 
%                      V_col = ceil(j/outlisersRAW);
%                      V_raw = mod(j,outlisersRAW);
%                      if V_raw == 0 
%                          V_raw = outlisersRAW;
%                      end
%                      
%                      [range_x_small , range_x_big , range_y_small , range_y_big] = deal(2);
%                      if V_col - 1 == 0
%                          range_y_small = 0;
%                      end
%                      if V_col - 2 == 0
%                          range_y_small = 1;
%                      end
%                      if V_col + 0 == outlisersCOL
%                          range_y_big = 0;
%                      end
%                      if V_col + 1 == outlisersCOL
%                          range_y_big = 1;
%                      end
%                      if V_raw - 1 == 0
%                          range_x_small = 0;
%                      end
%                      if V_raw - 2 == 0
%                          range_x_small = 1;
%                      end     
%                      if V_raw + 0 == outlisersRAW
%                          range_x_big = 0;
%                      end     
%                      if V_raw + 1 == outlisersRAW
%                          range_x_big = 1;
%                      end
%                 %      top_v_x = V_col - range_x_small;
%                 %      top_v_y = V_raw + range_y_big;
%                      top_v_x = V_raw - range_x_small;
%                      top_v_y = V_col + range_y_big;
%                      range_marix = [];
%                      for ii = 0:(range_x_small + range_x_big)
%                         for jj = 0:(range_y_small + range_y_big)
%                             temp_x = top_v_x + ii
%                             temp_y = top_v_y - jj
%                             temp_V = [V_copy(temp_x , temp_y , 1) V_copy(temp_x , temp_y , 2) V_copy(temp_x , temp_y , 3)];
%                             range_marix = [range_marix ; temp_V];
%                         end
%                      end
%                      data1=sortrows(range_marix);
%                     [dataUnique,r]=unique(data1,'rows');
%                     [dataUnique,r1]=unique(data1,'rows','last');
%                     a_result = r1-r+1;
%                     rang_marix_result = [dataUnique a_result];
%                     rang_marix_result = sortrows(rang_marix_result,-4);
%                      V(V_raw,V_col,1) = rang_marix_result(1,1);
%                      V(V_raw,V_col,2) = rang_marix_result(1,2);
%                      V(V_raw,V_col,3) = rang_marix_result(1,3);
%                    
%                  end
%              end
%           end
%     end
%   end
% end
% 
% for p = 1:2
% V_copy = [];
% V_copy = V;
% for i=1:ND
%    if halo(i) == 0 
%      V_col = ceil(i/outlisersRAW);
%      V_raw = mod(i,outlisersRAW);
%      if V_raw == 0 
%          V_raw = outlisersRAW;
%      end
%      [range_x_small , range_x_big , range_y_small , range_y_big] = deal(2);
%      if V_col - 1 == 0
%          range_y_small = 0;
%      end
%      if V_col - 2 == 0
%          range_y_small = 1;
%      end
%      if V_col + 0 == outlisersCOL
%          range_y_big = 0;
%      end
%      if V_col + 1 == outlisersCOL
%          range_y_big = 1;
%      end
%      if V_raw - 1 == 0
%          range_x_small = 0;
%      end
%      if V_raw - 2 == 0
%          range_x_small = 1;
%      end     
%      if V_raw + 0 == outlisersRAW
%          range_x_big = 0;
%      end     
%      if V_raw + 1 == outlisersRAW
%          range_x_big = 1;
%      end
%      top_v_x = V_raw - range_x_small;
%      top_v_y = V_col + range_y_big;
%      range_marix = [];
%      for ii = 0:(range_x_small + range_x_big)
%         for jj = 0:(range_y_small + range_y_big)
%             temp_x = top_v_x + ii
%             temp_y = top_v_y - jj
%             temp_V = [V_copy(temp_x , temp_y , 1) V_copy(temp_x , temp_y , 2) V_copy(temp_x , temp_y , 3)];
%             range_marix = [range_marix ; temp_V];
%         end
%      end
%      data1=sortrows(range_marix);
%     [dataUnique,r]=unique(data1,'rows');
%     [dataUnique,r1]=unique(data1,'rows','last');
%     a_result = r1-r+1;
%     rang_marix_result = [dataUnique a_result];
%     rang_marix_result = sortrows(rang_marix_result,-4);
%      V(V_raw,V_col,1) = rang_marix_result(1,1);
%      V(V_raw,V_col,2) = rang_marix_result(1,2);
%      V(V_raw,V_col,3) = rang_marix_result(1,3);
%      
%    end
% end
% end

figure, imshow(V)   % 显示分割结果
% figure, imshow(label2rgb(r3))   % 显示分割结果
toc
