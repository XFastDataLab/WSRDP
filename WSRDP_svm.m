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
    
   
    %��svm ��, ��x=[b;w]����||w||^2=w'*w= x'x-b^2    
    %��Ϊb��x�еĵ�һ������������ b=[1,0,0,...]*x,��ô||w||^2= x'x-  {[-1,0,0...]*x}'*{[-1,0,0...]*x}    
    %��c=[-1 0 0 0 0.... 0]', �������C=c*c';    
    %��||w||^2=w'*w= x'x-b^2  תΪ �� min ||w||^2 = min x'*(E - C)*x, 
    %��� f=[], Aeq=[], beq=[]
    [m_1,n]=size(data_1);
    [m_2,n]=size(data_2);
    H = eye(n);                  %����һ��nά��λ���� 
    c=zeros(n,1);
    c(1)=-1;                     %��������c
    H=H-c*c';                    %H=E-C=E-c*c'; 
    f=[];
    
    %���첻��ʽԼ��
    y_1=data_1(1,3);             %��һ��ı�ǩ 
    y_2=data_2(1,3);             %�ڶ���ı�ǩ 
    
    %������ѵ������ǰ���ϳ�����1,��x�еĵ�һ�����
    A_1=[ones(m_1, 1) data_1(:,1:2)];
    A_1=A_1*y_1;
    A_2=[ones(m_2, 1) data_2(:,1:2)];
    A_2=A_2*y_2;
    A=-[A_1; A_2];               %���������Լ���� �˸���������Ϊmatlab�滮�еĲ���ʽԼ���� Ax<=b,
                                 %�������е�Լ���������� y(i)(w'x+b>=1), 
    Aeq=[];                      %û�е�ʽԼ��
    beq=[];
    
    options =[];                 %�趨quadprog����ⷽ�������ڵ㷨����㷨�ȣ�������Ĭ�ϡ� options= optimset('MaxPCGIter',50);
    
    %����ʽ�ұ߳���Ϊ-1��ͬ��Ҫ��-1 , ���Ĳ���ʽԼ�����  -Ax<=-1 <==> Ax>=1 (�ȼ�)
    B = -ones(m_1+m_2,1);
    %ʹ��matlab �еĶ��ι滮����� x,���x�еĵ�һ�о��ǲ����е�b, ����ķ������� w
    
    [theta] = quadprog(H,[],A,B);
    
    %��ԭ�ɲ����е�w��b 
    b=theta(1);
    
    w=theta(2:length(theta));
    
    plot(data(:,1),data(:,2),'o');
    hold on
    


end

