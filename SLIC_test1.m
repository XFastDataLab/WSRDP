% 能够确定超像素位置的核心程序是  EnforceLabelConnectivity
% 11-123页码 
clc
clear
close all;
tic
img = imread('bee.jpg');
imshow(img)
title('original')
%设定超像素个数
K = 500;
%设定超像素紧凑系数
m_compactness = 100;

%%
img = DeSample(img,2);
img_size = size(img);
%转换到LAB色彩空间
cform = makecform('srgb2lab');       %rgb空间转换成lab空间 matlab自带的用法,Create color transformation structure
img_Lab = applycform(img, cform);    %rgb转换成lab空间
figure;
imshow(img_Lab)
title('img_lab')

%%
%得到超像素的LABXY种子点信息     
img_sz = img_size(1)*img_size(2);
superpixel_sz = img_sz/K;  % 每个超像素的像素点数
STEP = uint32(sqrt(superpixel_sz)); % 开方的边长
xstrips = uint32(img_size(2)/STEP);  % x方向 的超像素个数
ystrips = uint32(img_size(1)/STEP);  % y方向 的超像素个数
xstrips_adderr = double(img_size(2))/double(xstrips);  
ystrips_adderr = double(img_size(1))/double(ystrips);
numseeds = xstrips*ystrips;   % 实际的超像素个数
%种子点xy信息初始值为晶格中心亚像素坐标
%种子点Lab颜色信息为对应点最接近像素点的颜色通道值
kseedsx = zeros(numseeds, 1);
kseedsy = zeros(numseeds, 1);
kseedsl = zeros(numseeds, 1);
kseedsa = zeros(numseeds, 1);
kseedsb = zeros(numseeds, 1);

n = 1;
for y = 1: ystrips   % 第y个超像素
    for x = 1: xstrips   % 第 x 个超像素
        kseedsx(n, 1) = (double(x)-0.5)*xstrips_adderr; % 第x个种子点中心坐标，非准确描述
        kseedsy(n, 1) = (double(y)-0.5)*ystrips_adderr; % 第y个种子点中心坐标，非准确描述
        % 种子点中心对应LAB图上位置的 三通道值
        kseedsl(n, 1) = img_Lab(fix(kseedsy(n, 1)), fix(kseedsx(n, 1)), 1);  % fix 417.1296 变417
        kseedsa(n, 1) = img_Lab(fix(kseedsy(n, 1)), fix(kseedsx(n, 1)), 2); 
        kseedsb(n, 1) = img_Lab(fix(kseedsy(n, 1)), fix(kseedsx(n, 1)), 3);
        n = n+1;
    end
end

n = 1;
%根据种子点计算超像素分区
klabels = PerformSuperpixelSLIC(img_Lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, m_compactness);
%合并小的分区
[supmtrx,supmtry,nlabels] = EnforceLabelC(img_Lab, klabels, K);
% 这里的supmtrx,supmtry的每列分别是对应标签区域的全部x坐标和y坐标
function klabels = PerformSuperpixelSLIC(img_Lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, compactness)

[m_height, m_width, m_channel] = size(img_Lab);
[numseeds xxxxx]= size(kseedsl);
img_Lab = double(img_Lab);
%像素标签格式为(x, y) (行, 列)
klabels = zeros(m_height, m_width);
%聚类尺寸
clustersize = zeros(numseeds,1);
inv = zeros(numseeds,1);
sigmal = zeros(numseeds,1);
sigmaa = zeros(numseeds,1);
sigmab = zeros(numseeds,1);
sigmax = zeros(numseeds,1);
sigmay = zeros(numseeds,1);
invwt = 1/( (double(STEP)/double(compactness)) *(double(STEP)/double(compactness)) );
%invwt = double(compactness)/double(STEP);
distvec = 100000*ones(m_height, m_width);
numk = numseeds;
for itr = 1: 10   %迭代次数
    sigmal = zeros(numseeds, 1);
    sigmaa = zeros(numseeds, 1);
    sigmab = zeros(numseeds, 1);
    sigmax = zeros(numseeds, 1);
    sigmay = zeros(numseeds, 1);
    clustersize = zeros(numseeds, 1);
    inv = zeros(numseeds, 1);
    distvec = double(100000*ones(m_height, m_width));
    %根据当前种子点信息计算每一个像素的归属
    for n = 1: numk
        y1 = max(1, kseedsy(n, 1)-STEP);
        y2 = min(m_height, kseedsy(n, 1)+STEP);
        x1 = max(1, kseedsx(n, 1)-STEP);
        x2 = min(m_width, kseedsx(n, 1)+STEP);
        %按像素计算距离
        for y = y1: y2
            for x = x1: x2
                %dist_lab = abs(img_Lab(y, x, 1)-kseedsl(n))+abs(img_Lab(y, x, 2)-kseedsa(n))+abs(img_Lab(y, x, 3)-kseedsb(n));
                % lab图 点到种子点 定义距离差，判断相似度
                dist_lab = (img_Lab(y, x, 1)-kseedsl(n, 1))^2+(img_Lab(y, x, 2)-kseedsa(n, 1))^2+(img_Lab(y, x, 3)-kseedsb(n, 1))^2;
                % 改成平方啊 ！！！  @@@  ！！！  @@@   ！！！
                dist_xy = (double(y)-kseedsy(n, 1))*(double(y)-kseedsy(n, 1)) + (double(x)-kseedsx(n, 1))*(double(x)-kseedsx(n, 1));
                %dist_xy = abs(y-kseedsy(n)) + abs(x-kseedsx(n));
 
                %距离 = lab色彩空间距离 + 空间距离权重×空间距离
                dist = dist_lab + dist_xy*invwt;
                %在周围最多四个种子点中找到最相似的 标记后存入klabels
                %m = (y-1)*m_width+x;
                if (dist<distvec(y, x))
                    distvec(y, x) = dist;  % 不断变小
                    klabels(y, x) = n;  % n是标签，也就是值像素属于哪个种子点
                end
            end
        end
    end
    %完成一遍分类后，重新计算种子点位置 使其向梯度最小地方移动
    ind = 1;
    for r = 1: m_height
        for c = 1: m_width
            sigmal(klabels(r, c),1) = sigmal(klabels(r, c),1)+img_Lab(r, c, 1);  % 像素块内所有的通道值相加
            sigmaa(klabels(r, c),1) = sigmaa(klabels(r, c),1)+img_Lab(r, c, 2);
            sigmab(klabels(r, c),1) = sigmab(klabels(r, c),1)+img_Lab(r, c, 3);
            sigmax(klabels(r, c),1) = sigmax(klabels(r, c),1)+c;    % 像素块内所有的横坐标相加
            sigmay(klabels(r, c),1) = sigmay(klabels(r, c),1)+r;     % 像素块内所有的纵坐标相加
            clustersize(klabels(r, c),1) = clustersize(klabels(r, c),1)+1;   % 像素块内所有个数相加
        end
    end
    for m = 1: numseeds  % 第m个种子点
        if (clustersize(m, 1)<=0)
            clustersize(m, 1) = 1;
        end
        inv(m, 1) = 1/clustersize(m, 1);
    end
    function [supmtrx,supmtry,nlabels] = EnforceLabelC(img_Lab, labels, K)



for j = 1: m_height
    for k = 1: m_width
        %逐点寻找未标记的区域 小于0 才执行
        if (0>nlabels(m, n))
            %从第一个未标记的(m,n)起，确定一个新区域，用label标记该区域的起点，用蝶形前进
            nlabels(m, n) = label;
            %开始一个新的分割 记录起点坐标
            xvec(1, 1) = k;
            yvec(1, 1) = j;
            supmtrx(1, label) = k;
            supmtry(1, label) = j;
            %如果起点与某个已知区域相连 用adjlabel记录该区域编号 如果当前区域过小则与相邻区域合并
            for i = 1: 4
                x = xvec(1, 1)+dx(1, i);
                y = yvec(1, 1)+dy(1, i);
                if (x>0 && x<=m_width && y>0 && y<=m_height)
                    if (nlabels(y, x)>0)
                        adjlabel = nlabels(y, x);  % 一般是左临或上邻的标签
                    end
                end
            end