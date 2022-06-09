%% Ming Yan & Yewang Chen
% A Lightweight Weakly Supervised Learning Segmentation Algorithm for Imbalanced Image Based on Rotation Density Peaks
% Knowledge-based System, Volume 244, 2022, 108513
% CITE:
% @article{yan2022lightweight,
%   title={A lightweight weakly supervised learning segmentation algorithm for imbalanced image based on rotation density peaks},
%   author={Yan, Ming and Chen, Yewang and Chen, Yi and Zeng, Guoyao and Hu, Xiaoliang and Du, Jixiang},
%   journal={Knowledge-Based Systems},
%   volume={244},
%   pages={108513},
%   year={2022},
%   publisher={Elsevier}
% }
%% Support vector training with weak information. @run on matlab r2012a

clear all

% load('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\data\BSR_test\test_135069\135069_40percent_dc0_02974_rho.mat');
% load('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\data\BSR_test\test_135069\135069_40percent_dc0_02974_delta.mat');

I_test = imread('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\data\BSR_test\135069.jpg'); 
% Whether to use superpixel acceleration, defaults 1. TIPS: This is a non-essential parameter
superpix = 0;
% non-dependent parameter, default 1(lab), 0.02(hsv), 0.01(rgb)
dc = 0.01;
% Which feature extraction to use:1(lab), 2(hsv), 3(rgb)
feaExtra = 3;

% Image Encoder
[Lab_mean, ~] = WSRDP_ImgaeEncoder(I_test, superpix, feaExtra);
[rho, delta, ~] = WSRDP_FastDistence(Lab_mean, dc);


% According to the sparsity of the image, select the correct density peaks.
ids = [4798
       10654
       12226];
[rho,PS] = mapminmax(rho,0,1);
[delta,PS] = mapminmax(delta,0,1);
theta = pi/(7);
T=[cos(theta) -sin(theta);
   sin(theta) cos(theta)];
deta_gamma_delta=[rho; delta];
new_data=T*deta_gamma_delta;
new_data=new_data';
new_rho=new_data(:,1);
new_delta=new_data(:,2);
X = [new_rho new_delta];

plot(new_rho,new_delta,'o');
hold on

% Running this statement can directly get the subscript of the "brushedData"
% ids = WSRDP_findIdx(brushedData);

Y = ones(length(new_rho),1);
for i = 1: length(ids)
    Y(ids(i)) = -1;
end
data = [X Y];

% Determining decision curves in svm via quadratic hyperplanes.
[w, b] = WSRDP_svm(data);

% w = [-0.192563065647279 -6.32691631743245];
% b = 4.27931006272744;

% Drawing decision curves
z_1=0:0.001:0.8;
z_2=(w(1)/w(2)+0.1)*z_1-(b/w(2))-0.02;
z_1=sqrt(z_1);

svm_test_a = (w(1)/w(2)); %[0.0304355322539533]
svm_test_c = -(b/w(2))-0.02; %[0.656365838905871]

hold on
plot(z_1,z_2,'r');
plot(-z_1,z_2,'r'); 
plot( X(:,1), X(:,2),'.','MarkerSize',15);
xlabel('\rho');
ylabel('\delta');



%% Training weak information by SVR
% Solve multi-prediction tasks using single prediction
SVR_X = [];
SVR_Y = [];
% Two SVM regression models were cross-validated using 5-fold cross-validation.
% Specifies to use the default linear kernel for training and a Gaussian kernel for another model.
MdlLin = fitrsvm(SVR_X,SVR_Y,'Standardize',true,'KFold',5)
MdlGau = fitrsvm(SVR_X,SVR_Y,'Standardize',true,'KFold',5,'KernelFunction','gaussian')
mseLin = kfoldLoss(MdlLin)
mseGau = kfoldLoss(MdlGau)

% Use fitrsvm to automatically optimize hyperparameters.
%Find hyperparameters that reduce cross-validation loss by a factor of five by using automatic hyperparameter optimization.
rng default
Mdl = fitrsvm(SVR_X,SVR_Y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'))
fit = predict(Mdl,7);


%% Add weak information to segment images in Rotation DPeaks Clustering
% Just run this section with the weak information trained in the paper.
clear all
tic
% input test set image
I = imread('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\data\BSR_test\238011.jpg'); 
% Whether this image is an unbalanced image, defaults 1. TIPS: This is a non-essential parameter
object_parameter = 1;
% Directly use the parameters trained in the paper.
% We train on small objects in different unbalanced images and obtain two applicable parameters.
% When the small object to be identified is relatively large, "object_parameter = 1" is used.
% When the small object is relatively small, "object_parameter = 0" is used.
if object_parameter 
    a=1.03043553225395;%a samll->broad       ,defult=1
    c=0.51365838905871;
    m=-0 ;
    theta=pi/(7);
else
    a=3.09124783142895;
    c=0.30178297148251;
    m=-0 ;
    theta=pi/(7);
end

% Whether to use superpixel acceleration, defaults 1. TIPS: This is a non-essential parameter
superpix = 1;
% non-dependent parameter, default 1(lab), 0.02(hsv), 0.01(rgb)
dc = 1;
% Which feature extraction to use:1(lab), 2(hsv), 3(rgb)
feaExtra = 1;

% Image Encoder
[Lab_mean, originalRAW, originalCOL, NB_Label, point_line, newRAW, newCOL] = WSRDP_ImgaeEncoder(I, superpix, feaExtra);

% main program
[cl, ids] = WSRDP_Rotation_DPeaks(Lab_mean,dc,a,c,m,theta);

% Image Decoder
if superpix == 1
    r2 = zeros(originalRAW,originalCOL);
    w = 1;
    for k=1:NB_Label
        temp = point_line{k,1};
        if isempty(temp)
            continue;
        end
        for i=1:length(temp(:,1))
            r2(temp(i,4),temp(i,5)) = cl(w);
        end
        w = w + 1;
    end
    figure, imshow(label2rgb(r2,colormap(lines)))
else
    r3 = reshape(cl, (newRAW-1), newCOL);
    V = label2rgb(r3,colormap(lines));
    figure, imshow(V)
end

toc