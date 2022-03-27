%################################################################################
% Modified from original density peaks
% Author : Ming Yan, Yewang Chen 
% Email  : 19014083027@stu.hqu.edu.cn,ywchen@hqu.edu.cn
% Version:1.0
% Date   :2020/5/11             
% College of Computer Science and Technologyï¼?Huaqiao University, Xiamen, China
% Copyright@hqu.edu.cn
%################################################################################
function [rho, ordrho, delta,ND,nneigh]= dpeak(distances,percent)    
    xx=distances;
    ND=max(xx(:,2));
    NL=max(xx(:,1));
    if (NL>ND)
      ND=NL;
    end
    N=size(xx,1);
    for i=1:ND
      for j=1:ND
        dist(i,j)=0;
      end
    end
    for i=1:N
      ii=xx(i,1);
      jj=xx(i,2);
      dist(ii,jj)=xx(i,3);
      dist(jj,ii)=xx(i,3);
    end
    fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);%average percentage of neighbours (hard coded)
    position=round(N*percent/100);
    sda=sort(xx(:,3));
    dc=sda(position);
    fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);%Computing Rho with gaussian kernel of radius
    for i=1:ND
      rho(i)=0.;
    end
    %
    % Gaussian kernel
    %
    for i=1:ND-1
      for j=i+1:ND
         rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
         rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
      end
    end
    %
    % "Cut off" kernel
    %
    %for i=1:ND-1
    %  for j=i+1:ND
    %    if (dist(i,j)<dc)
    %       rho(i)=rho(i)+1.;
    %       rho(j)=rho(j)+1.;
    %    end
    %  end
    %end

    maxd=max(max(dist));

    [~,ordrho]=sort(rho,'descend');
    delta(ordrho(1))=-1.;
    nneigh(ordrho(1))=0;

    for ii=2:ND
       delta(ordrho(ii))=maxd;
       for jj=1:ii-1
         if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
            delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
            nneigh(ordrho(ii))=ordrho(jj);
         end
       end
    end    
end