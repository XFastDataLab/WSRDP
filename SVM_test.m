clc
clear all
close all

% load carsmall
% rng 'default'  % For reproducibility
% X = [Horsepower Weight];
% Y = MPG;

X = [7; 7; 7; 7];
Y = [0.6; 0.61; 0.72; 0.6];

% 使用5-fold交叉验证对两个SVM回归模型进行交叉验证。 
% 对于这两种模型，请指定以标准化预测变量。 对于其中一个模型，
% 指定使用默认线性核进行训练，而对于另一个模型，则指定使用高斯核。
MdlLin = fitrsvm(X,Y,'Standardize',true,'KFold',5)
MdlGau = fitrsvm(X,Y,'Standardize',true,'KFold',5,'KernelFunction','gaussian')
mseLin = kfoldLoss(MdlLin)
mseGau = kfoldLoss(MdlGau)

% 使用fitrsvm自动优化超参数。通过使用自动超参数优化，找到使交叉验证损失减少五倍的超参数。
rng default
Mdl = fitrsvm(X,Y,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'))
% 预测
fit = predict(Mdl,7)
plot(7,fit,'.')


load('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\data\BSR_test\test_135069\135069_40percent_dc0_02974_rho.mat');
load('D:\PostgraduateStudy\Rotation-DPeak\New_Segmentation_Image\data\BSR_test\test_135069\135069_40percent_dc0_02974_delta.mat');
ids = [5062,10655,12096];
[rho,PS] = mapminmax(rho,0,1);
[delta,PS] = mapminmax(delta,0,1);
theta=pi/(7)
T=[cos(theta) -sin(theta);
   sin(theta) cos(theta)];
deta_gamma_delta=[rho; delta];
new_data=T*deta_gamma_delta;
new_data=new_data';
new_rho=new_data(:,1);
new_delta=new_data(:,2);
X = [new_rho new_delta];
Y = ones(length(new_rho),1);
for i = 1: length(ids)
    Y(ids(i)) = -1;
end

X_q = X;
X_q(:,1) = X_q(:,1).^2;

model = svmtrain(Y, X_q, '-t 0 -c 1');
[predict_label, accuracy, dec_values] = svmpredict(y, x, model);

%支持向量索引(Support Vectors Index)
SVs_idx = model.sv_indices;

%支持向量特征属性和类别属性
x_SVs = X_q(SVs_idx,:);% or use: SVs=full(model.SVs);
y_SVs = Y(SVs_idx);

%求平面w^T x + b = 0的法向量w
alpha_SVs = model.sv_coef;%实际是a_i*y_i
w = sum(diag(alpha_SVs)*x_SVs)';%即西瓜书公式(6.9)

%求平面w^T x + b = 0的偏移项b
%由于是软件隔支持向量机，所以先找出正好在最大间隔边界上的支持向量
SVs_on = (abs(alpha_SVs)<1);%C=1 by parameter '-c 1'
y_SVs_on = y_SVs(SVs_on,:);
x_SVs_on = x_SVs(SVs_on,:);
%理论上可选取任意在最大间隔边界上的支持向量通过求解西瓜书式(6.17)获得b
b_temp = zeros(1,sum(SVs_on));%所有的b
for idx=1:sum(SVs_on)
    b_temp(idx) = 1/y_SVs_on(idx)-x_SVs_on(idx,:)*w;
end
b = mean(b_temp);%更鲁棒的做法是使用所有支持向量求解的平均值

%将手动计算出的偏移项b与svmtrain给出的偏移项b对比
b_model = -model.rho;%model中的rho为-b
b-b_model

%将手动计算出的决策值与svmpredict输出的决策值对比
%决策值f(x)=w^T x + b
f_x = X_q * w + b;
sum(abs(f_x-dec_values))

w = [-2.426778292208671 -13.567743836922482];
b = 9.124482775845749;

z_1=0:0.001:0.8;
z_2=(w(1)/w(2))*z_1-(b/w(2))-0.02;
z_1=sqrt(z_1);

hold on
plot(z_1,z_2,'r');
plot(-z_1,z_2,'r'); 
% plot( X_q(:,1), X_q(:,2),'.','MarkerSize',15);
plot( X(:,1), X(:,2),'.','MarkerSize',15);
xlabel('\rho');
ylabel('\delta');

z_1=-0.05:0.001:0.8;
z_2=(w(1)/w(2))*z_1-(b/w(2))-0.02;
plot( X_q(:,1), X_q(:,2),'.','MarkerSize',15);
hold on
plot(z_1,z_2,'r');
xlabel('\rho^2');
ylabel('\delta');
box off

xlabel('\rho');
ylabel('\rho^2');

z_1=0:0.001:0.8;
z_2=(w(1)/w(2))*z_1-(b/w(2))-0.02;
z_1=sqrt(z_1);

hold on
plot(z_1,z_2,'r');
plot(-z_1,z_2,'r'); 
% plot( X_q(:,1), X_q(:,2),'.','MarkerSize',15);
plot( X(:,1), X(:,2),'.','MarkerSize',15);
axis off

load ionosphere
rng default
Mdl = fitcsvm(X,Y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))

SVMModel = fitcsvm(X,Y);
w = SVMModel.W(1);
b = SVMModel.Bias;


xtest=-1:0.001:1;
ytest=w*xtest+ b;

hold on
figure(1)
plot(new_rho,new_delta,'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
plot(xtest,ytest)

hold on
plot(z_1,z_2,'r.'); 
plot(-z_1,z_2,'r.');

figure(2)
plot(xtest,ytest)

SVINDEX = model.sv_indices;
SVINDEX = [];
for i=1:length(new_rho(:,1))
    if SVIndex(i) == 1
        SVINDEX = [SVINDEX ; i];
    end 
end

for i=1:length(SVINDEX)
    hold on
    figure(1)
    plot(X_q(SVINDEX(i),1),X_q(SVINDEX(i),2),'o','MarkerSize',8,'MarkerEdgeColor','r');
end

