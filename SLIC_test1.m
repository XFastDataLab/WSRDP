% �ܹ�ȷ��������λ�õĺ��ĳ�����  EnforceLabelConnectivity
% 11-123ҳ�� 
clc
clear
close all;
tic
img = imread('bee.jpg');
imshow(img)
title('original')
%�趨�����ظ���
K = 500;
%�趨�����ؽ���ϵ��
m_compactness = 100;

%%
img = DeSample(img,2);
img_size = size(img);
%ת����LABɫ�ʿռ�
cform = makecform('srgb2lab');       %rgb�ռ�ת����lab�ռ� matlab�Դ����÷�,Create color transformation structure
img_Lab = applycform(img, cform);    %rgbת����lab�ռ�
figure;
imshow(img_Lab)
title('img_lab')

%%
%�õ������ص�LABXY���ӵ���Ϣ     
img_sz = img_size(1)*img_size(2);
superpixel_sz = img_sz/K;  % ÿ�������ص����ص���
STEP = uint32(sqrt(superpixel_sz)); % �����ı߳�
xstrips = uint32(img_size(2)/STEP);  % x���� �ĳ����ظ���
ystrips = uint32(img_size(1)/STEP);  % y���� �ĳ����ظ���
xstrips_adderr = double(img_size(2))/double(xstrips);  
ystrips_adderr = double(img_size(1))/double(ystrips);
numseeds = xstrips*ystrips;   % ʵ�ʵĳ����ظ���
%���ӵ�xy��Ϣ��ʼֵΪ������������������
%���ӵ�Lab��ɫ��ϢΪ��Ӧ����ӽ����ص����ɫͨ��ֵ
kseedsx = zeros(numseeds, 1);
kseedsy = zeros(numseeds, 1);
kseedsl = zeros(numseeds, 1);
kseedsa = zeros(numseeds, 1);
kseedsb = zeros(numseeds, 1);

n = 1;
for y = 1: ystrips   % ��y��������
    for x = 1: xstrips   % �� x ��������
        kseedsx(n, 1) = (double(x)-0.5)*xstrips_adderr; % ��x�����ӵ��������꣬��׼ȷ����
        kseedsy(n, 1) = (double(y)-0.5)*ystrips_adderr; % ��y�����ӵ��������꣬��׼ȷ����
        % ���ӵ����Ķ�ӦLABͼ��λ�õ� ��ͨ��ֵ
        kseedsl(n, 1) = img_Lab(fix(kseedsy(n, 1)), fix(kseedsx(n, 1)), 1);  % fix 417.1296 ��417
        kseedsa(n, 1) = img_Lab(fix(kseedsy(n, 1)), fix(kseedsx(n, 1)), 2); 
        kseedsb(n, 1) = img_Lab(fix(kseedsy(n, 1)), fix(kseedsx(n, 1)), 3);
        n = n+1;
    end
end

n = 1;
%�������ӵ���㳬���ط���
klabels = PerformSuperpixelSLIC(img_Lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, m_compactness);
%�ϲ�С�ķ���
[supmtrx,supmtry,nlabels] = EnforceLabelC(img_Lab, klabels, K);
% �����supmtrx,supmtry��ÿ�зֱ��Ƕ�Ӧ��ǩ�����ȫ��x�����y����
function klabels = PerformSuperpixelSLIC(img_Lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, compactness)

[m_height, m_width, m_channel] = size(img_Lab);
[numseeds xxxxx]= size(kseedsl);
img_Lab = double(img_Lab);
%���ر�ǩ��ʽΪ(x, y) (��, ��)
klabels = zeros(m_height, m_width);
%����ߴ�
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
for itr = 1: 10   %��������
    sigmal = zeros(numseeds, 1);
    sigmaa = zeros(numseeds, 1);
    sigmab = zeros(numseeds, 1);
    sigmax = zeros(numseeds, 1);
    sigmay = zeros(numseeds, 1);
    clustersize = zeros(numseeds, 1);
    inv = zeros(numseeds, 1);
    distvec = double(100000*ones(m_height, m_width));
    %���ݵ�ǰ���ӵ���Ϣ����ÿһ�����صĹ���
    for n = 1: numk
        y1 = max(1, kseedsy(n, 1)-STEP);
        y2 = min(m_height, kseedsy(n, 1)+STEP);
        x1 = max(1, kseedsx(n, 1)-STEP);
        x2 = min(m_width, kseedsx(n, 1)+STEP);
        %�����ؼ������
        for y = y1: y2
            for x = x1: x2
                %dist_lab = abs(img_Lab(y, x, 1)-kseedsl(n))+abs(img_Lab(y, x, 2)-kseedsa(n))+abs(img_Lab(y, x, 3)-kseedsb(n));
                % labͼ �㵽���ӵ� ��������ж����ƶ�
                dist_lab = (img_Lab(y, x, 1)-kseedsl(n, 1))^2+(img_Lab(y, x, 2)-kseedsa(n, 1))^2+(img_Lab(y, x, 3)-kseedsb(n, 1))^2;
                % �ĳ�ƽ���� ������  @@@  ������  @@@   ������
                dist_xy = (double(y)-kseedsy(n, 1))*(double(y)-kseedsy(n, 1)) + (double(x)-kseedsx(n, 1))*(double(x)-kseedsx(n, 1));
                %dist_xy = abs(y-kseedsy(n)) + abs(x-kseedsx(n));
 
                %���� = labɫ�ʿռ���� + �ռ����Ȩ�ء��ռ����
                dist = dist_lab + dist_xy*invwt;
                %����Χ����ĸ����ӵ����ҵ������Ƶ� ��Ǻ����klabels
                %m = (y-1)*m_width+x;
                if (dist<distvec(y, x))
                    distvec(y, x) = dist;  % ���ϱ�С
                    klabels(y, x) = n;  % n�Ǳ�ǩ��Ҳ����ֵ���������ĸ����ӵ�
                end
            end
        end
    end
    %���һ���������¼������ӵ�λ�� ʹ�����ݶ���С�ط��ƶ�
    ind = 1;
    for r = 1: m_height
        for c = 1: m_width
            sigmal(klabels(r, c),1) = sigmal(klabels(r, c),1)+img_Lab(r, c, 1);  % ���ؿ������е�ͨ��ֵ���
            sigmaa(klabels(r, c),1) = sigmaa(klabels(r, c),1)+img_Lab(r, c, 2);
            sigmab(klabels(r, c),1) = sigmab(klabels(r, c),1)+img_Lab(r, c, 3);
            sigmax(klabels(r, c),1) = sigmax(klabels(r, c),1)+c;    % ���ؿ������еĺ��������
            sigmay(klabels(r, c),1) = sigmay(klabels(r, c),1)+r;     % ���ؿ������е����������
            clustersize(klabels(r, c),1) = clustersize(klabels(r, c),1)+1;   % ���ؿ������и������
        end
    end
    for m = 1: numseeds  % ��m�����ӵ�
        if (clustersize(m, 1)<=0)
            clustersize(m, 1) = 1;
        end
        inv(m, 1) = 1/clustersize(m, 1);
    end
    function [supmtrx,supmtry,nlabels] = EnforceLabelC(img_Lab, labels, K)



for j = 1: m_height
    for k = 1: m_width
        %���Ѱ��δ��ǵ����� С��0 ��ִ��
        if (0>nlabels(m, n))
            %�ӵ�һ��δ��ǵ�(m,n)��ȷ��һ����������label��Ǹ��������㣬�õ���ǰ��
            nlabels(m, n) = label;
            %��ʼһ���µķָ� ��¼�������
            xvec(1, 1) = k;
            yvec(1, 1) = j;
            supmtrx(1, label) = k;
            supmtry(1, label) = j;
            %��������ĳ����֪�������� ��adjlabel��¼�������� �����ǰ�����С������������ϲ�
            for i = 1: 4
                x = xvec(1, 1)+dx(1, i);
                y = yvec(1, 1)+dy(1, i);
                if (x>0 && x<=m_width && y>0 && y<=m_height)
                    if (nlabels(y, x)>0)
                        adjlabel = nlabels(y, x);  % һ�������ٻ����ڵı�ǩ
                    end
                end
            end