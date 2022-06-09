%author ywchen@hqu.edu.cn
%Use quadratic programming to complete svm, do linear classification of data
function  [w, b] = WSRDP_svm(data)
    newdata=data;
    newdata(:,1)=newdata(:,1).^2;
    idx=find(newdata(:,3)==-1);
    data_1= newdata(idx,:);
    newdata(idx,:)=[];
    data_2=newdata;
    
    %The basic form of quadratic programming is as follows:
    % min 0.5 x'*H*x + f'*x    
    % st.    Ax<=b
    %      Aeq*x=beq    
    
   
    %在svm 中, 令x=[b;w]，则||w||^2=w'*w= x'x-b^2    
    %因为b是x中的第一个分量，所以 b=[1,0,0,...]*x,那么||w||^2= x'x-  {[-1,0,0...]*x}'*{[-1,0,0...]*x}    
    %令c=[-1 0 0 0 0.... 0]', 再令矩阵C=c*c';    
    %则||w||^2=w'*w= x'x-b^2  转为 求 min ||w||^2 = min x'*(E - C)*x, 
    %因而 f=[], Aeq=[], beq=[]
    [m_1,n]=size(data_1);
    [m_2,n]=size(data_2);
    H = eye(n);                  %构造一个n维单位矩阵 
    c=zeros(n,1);
    c(1)=-1;                     %构造向量c
    H=H-c*c';                    %H=E-C=E-c*c'; 
    f=[];
    
    %构造不等式约束
    y_1=data_1(1,3);             %第一类的标签 
    y_2=data_2(1,3);             %第二类的标签 
    
    %在所有训练数据前补上常数项1,跟x中的第一项对齐
    A_1=[ones(m_1, 1) data_1(:,1:2)];
    A_1=A_1*y_1;
    A_2=[ones(m_2, 1) data_2(:,1:2)];
    A_2=A_2*y_2;
    A=-[A_1; A_2];               %构造好线性约束， 乘个负号是因为matlab规划中的不等式约束是 Ax<=b,
                                 %而材料中的约束是条件是 y(i)(w'x+b>=1), 
    Aeq=[];                      %没有等式约束
    beq=[];
    
    options =[];                 %设定quadprog的求解方法，有内点法，外点法等，这里用默认。 options= optimset('MaxPCGIter',50);
    
    %不等式右边常量为-1，同样要乘-1 , 最后的不等式约束变成  -Ax<=-1 <==> Ax>=1 (等价)
    B = -ones(m_1+m_2,1);
    %使用matlab 中的二次规划，算出 x,这个x中的第一列就是材料中的b, 后面的分量构成 w
    
    [theta] = quadprog(H,[],A,B);
    
    %还原成材料中的w和b 
    b=theta(1);
    
    w=theta(2:length(theta));
    
    plot(data(:,1),data(:,2),'o');
    hold on
    


end

