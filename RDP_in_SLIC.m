
clear all;
tic
load('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\data\BSR_test\test_3063\SLIC_223004_Label_15.mat');
I = imread('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\data\BSR_test\223004.jpg'); 
figure, imshow(I);    % 读入分割目标
gamma_num = 3;
data_sparse = 1;
originalRAW = 321;
originalCOL = 481;

% A = reshape(I(:, :, 1), originalRAW*originalCOL, 1);    
% B = reshape(I(:, :, 2), originalRAW*originalCOL, 1);
% C = reshape(I(:, :, 3), originalRAW*originalCOL, 1);
% A_double = im2double(A); 
% B_double = im2double(B);
% C_double = im2double(C);

%计算HSV
% I_h = rgb2hsv(I);
% A_h = reshape(I_h(:, :, 1), originalRAW*originalCOL, 1); 
% B_h = reshape(I_h(:, :, 2), originalRAW*originalCOL, 1); 
% C_h = reshape(I_h(:, :, 3), originalRAW*originalCOL, 1); 

%计算Lab
I_l = rgb2lab(I);
A_l = reshape(I_l(:, :, 1), originalRAW*originalCOL, 1); 
B_l = reshape(I_l(:, :, 2), originalRAW*originalCOL, 1); 
C_l = reshape(I_l(:, :, 3), originalRAW*originalCOL, 1); 

NB_Label = Label(originalRAW,originalCOL);
point_line = cell(NB_Label,1);

for i=1:originalRAW
    for j=1:originalCOL
        point_line{Label(i,j),1} = [point_line{Label(i,j),1}; I_l(i,j,1) I_l(i,j,2) I_l(i,j,3) i j];
    end
end

Lab_mean = [];
for k=1:NB_Label
    temp = point_line{k,1};
    if isempty(temp)
        continue;
    end
    Lab_mean = [Lab_mean ; mean(temp(:,1:3),1)];
end

% c2 = kmeans(Lab_mean, gamma_num); 
% minpts = 100;
% epsilon = 1.02;
% c2 = DBSCAN_ym(Lab_mean,epsilon,minpts);

dc = 1;
% c2 = DPeaks_original(Lab_mean,gamma_num,dc);
c2 = Rotation_DPeaks(Lab_mean,dc,data_sparse);


r2 = zeros(originalRAW,originalCOL);

w = 1;
for k=1:NB_Label
    temp = point_line{k,1};
    if isempty(temp)
        continue;
    end
    for i=1:length(temp(:,1))
        r2(temp(i,4),temp(i,5)) = c2(w);
    end
    w = w + 1;
end

figure, imshow(label2rgb(r2,colormap(lines)))

toc
