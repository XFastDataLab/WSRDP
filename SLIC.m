function Label=SLIC(img,s,errTh,wDs)

m=size(img,1);
n=size(img,2);
 
%% Calculate the coordinates of the vertices and the center of the grid
h=floor(m/s);
w=floor(n/s);
rowR=floor((m-h*s)/2); 
colR=floor((n-w*s)/2);
rowStart=(rowR+1):s:(m-s+1);
rowStart(1)=1;
rowEnd=rowStart+s;
rowEnd(1)=rowR+s;
rowEnd(end)=m;
colStart=(colR+1):s:(n-s+1);
colStart(1)=1;
colEnd=colStart+s;
colEnd(1)=colR+s;
colEnd(end)=n;
rowC=floor((rowStart+rowEnd-1)/2);
colC=floor((colStart+colEnd-1)/2);
% Show division results
temp=zeros(m,n);
temp(rowStart,:)=1;
temp(:,colStart)=1;
for i=1:h
    for j=1:w
        temp(rowC(i),colC(j))=1;
    end
end

img=double(img)/255;
r=img(:,:,1);
g=img(:,:,2);
b=img(:,:,3);
Y=0.299 * r + 0.587 * g + 0.114 * b;
 
f1=fspecial('sobel');
f2=f1';
gx=imfilter(Y,f1);
gy=imfilter(Y,f2);
G=sqrt(gx.^2+gy.^2); 

rowC_std=repmat(rowC',[1,w]);
colC_std=repmat(colC,[h,1]);
rowC=rowC_std;
colC=colC_std;
for i=1:h
    for j=1:w
        block=G(rowC(i,j)-1:rowC(i,j)+1,colC(i,j)-1:colC(i,j)+1);
        [minVal,idxArr]=min(block(:));
        jOffset=floor((idxArr(1)+2)/3);
        iOffset=idxArr(1)-3*(jOffset-1);
        rowC(i,j)=rowC(i,j)+iOffset;
        colC(i,j)=colC(i,j)+jOffset;
    end
end
 

Label=zeros(m,n)-1;
dis=Inf*ones(m,n);
M=reshape(img,m*n,size(img,3)); 
% Joint gamut value and empty gamut value
colorC=zeros(h,w,size(img,3));
for i=1:h
    for j=1:w
        colorC(i,j,:)=img(rowC(i),colC(j),:);
    end
end
uniMat=cat(3,colorC,rowC,colC);
uniMat=reshape(uniMat,h*w,size(img,3)+2);
iter=1;
while(1)
    uniMat_old=uniMat;
%     rowC_old=rowC;
%     colC_old=colC;
    for k=1:h*w
        c=floor((k-1)/h)+1;
        r=k-h*(c-1);
        rowCidx=rowC(r,c);
        colCidx=colC(r,c); 
        rowStart=max(1,rowC_std(r,c)-s);
        rowEnd=min(m,rowC_std(r,c)+s-1);
        colStart=max(1,colC_std(r,c)-s);
        colEnd=min(n,colC_std(r,c)+s);
%         colorC=uniMat(k,1:size(img,3));
        colorC=M((colCidx-1)*m+rowCidx,:);
        for i=rowStart:rowEnd
            for j=colStart:colEnd
                colorCur=M((j-1)*m+i,:);
                dc=norm(colorC-colorCur);
                ds=norm([i-rowCidx,j-colCidx]);
                d=dc^2+wDs*(ds/s)^2;
                if d<dis(i,j)
                    dis(i,j)=d;
                    Label(i,j)=k;
                end
            end
        end
    end
    
    %Display clustering results
    temp=mod(Label,20)+1;
    colorC=zeros(h,w,size(img,3));
    for k=1:h*w
        num=0;
        sumColor=zeros(1,size(img,3));    
        sumR=0;
        sumC=0;
        c=floor((k-1)/h)+1;
        r=k-h*(c-1);
        rowCidx=rowC_std(r,c);
        colCidx=colC_std(r,c);
        rowStart=max(1,rowCidx-s);
        rowEnd=min(m,rowCidx+s-1);
        colStart=max(1,colCidx-s);
        colEnd=min(n,colCidx+s);
        
        for row=rowStart:rowEnd
            for col=colStart:colEnd
                if Label(row,col)==k
                    num=num+1;
                    sumR=sumR+row;
                    sumC=sumC+col;
                    color=reshape(img(row,col,:),1,size(img,3));
                    sumColor=sumColor+color;
                end
            end
        end
        colorC(r,c,:)=sumColor/num;
        rowC(r,c)=round(sumR/num);
        colC(r,c)=round(sumC/num);
    end
    uniMat=cat(3,colorC,rowC,colC);
    uniMat=reshape(uniMat,h*w,size(img,3)+2);
    diff=uniMat-uniMat_old;
    diff(:,1:2)=sqrt(wDs)*diff(:,1:2)/s;
    err=norm(diff)/sqrt(h*w);
    if err<errTh 
        break;
    end
end
 
%% Post-processing, assign the labels of small connected domains according to the principle of the maximum number of boundary contact points
for k=1:h*w
    c=floor((k-1)/h)+1;
    r=k-h*(c-1);
    rowCidx=rowC_std(r,c);
    colCidx=colC_std(r,c);
    rowStart=max(1,rowCidx-s);
    rowEnd=min(m,rowCidx+s-1);
    colStart=max(1,colCidx-s);
    colEnd=min(n,colCidx+s);
    block=Label(rowStart:rowEnd,colStart:colEnd);
    block(block~=k)=0;
    block(block==k)=1;
    label=bwlabel(block);
    szlabel=max(label(:)); 
    bh=rowEnd-rowStart+1;
    bw=colEnd-colStart+1;  
    
    if szlabel<2  
        continue;
    end
    
    labelC=label(rowCidx-rowStart+1,colCidx-colStart+1); 
    top=max(1,rowStart-1);
    bottom=min(m,rowEnd+1);
    left=max(1,colStart-1);
    right=min(n,colEnd+1);
    for i=1:szlabel 
        if i==labelC 
            continue;
        end
        marker=zeros(bottom-top+1,right-left+1);
        bw=label;
        bw(bw~=i)=0;
        bw(bw==i)=1; 
        contourBW=bwperim(bw); 
        %             figure,imshow(contourBW);
        idxArr=find(double(contourBW)==1);
        labelArr=zeros(4*length(idxArr),1);  
        num=0;
        for idx=1:size(idxArr) 
            bc=floor((idxArr(idx)-1)/bh)+1;
            br=idxArr(idx)-bh*(bc-1); 
            row=br+rowStart-1;
            col=bc+colStart-1; 
            rc=[row-1,col;...
                row+1,col;...
                row,col-1;...
                row,col+1];
            for p=1:4
                row=rc(p,1);
                col=rc(p,2);
                
                if ~(row>=1 && row<=m && col>=1 && col<=n && Label(row,col)~=k)
                    continue;
                end
                
                if marker(row-top+1,col-left+1)==0 
                    marker(row-top+1,col-left+1)=1;
                    num=num+1;
                    labelArr(num)=Label(row,col);
                end
            end
        end
        
        labelArr(find(labelArr==0))=[];
        uniqueLabel=unique(labelArr);
        numArr=zeros(length(uniqueLabel),1);
        for p=1:length(uniqueLabel)
            idx=find(labelArr==uniqueLabel(p));
            numArr(p)=length(idx);
        end
        idx=find(numArr==max(numArr));
        maxnumLabel=uniqueLabel(idx(1));
        
        for row=rowStart:rowEnd
            for col=colStart:colEnd
                if bw(row-rowStart+1,col-colStart+1)==0
                    continue;
                end
                Label(row,col)=maxnumLabel;
            end
        end
    end
end
 
% Display the clustering results after connected domain processing
% temp=mod(Label,20)+1;
% figure;
% imagesc(label2rgb(temp-1,'jet','w','shuffle')) ;
% axis image ; axis off ;