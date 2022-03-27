
points = load('D:\PostgraduateStudy\Rotation-DPeak\Code\RotationDP_data\data5.mat');

[x,y]=size(points);
[idx, dist] = knnsearch(points,points,'dist','Euclidean','k',1160);

%º∆À„ ˝¡ø
num= (x*x-x)/2;
distances=zeros(num,3);

distMat= pdist2(points,points);
k=1;
loc=1;
for i=1:x-1
    distances(loc:loc+(x-k)-1,1)=k*ones(x-k,1);
    distances(loc:loc+(x-k)-1,2)=(k+1:1:x);
    distances(loc:loc+(x-k)-1,3)=distMat(k,k+1:x)';
    loc=loc+(x-k);
    k=k+1;
end

% save 'ym_semicircle_distances' distances;
%save 'new_point_distMat' distMat;

