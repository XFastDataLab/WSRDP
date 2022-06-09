function [Lab_mean, originalRAW, originalCOL, NB_Label, point_line, newRAW, newCOL] = WSRDP_ImgaeEncoder(I, superpix, feaExtra)
    originalRAW = length(I(:,1,1));
    originalCOL = length(I(1,:,1));

    % Image Feature Encoder
    if feaExtra == 1  % extract image Lab features
        I_Extract = rgb2lab(I);
        if superpix == 0
           A_Extract = reshape(I_Extract(:, :, 1), originalRAW*originalCOL, 1); 
           B_Extract = reshape(I_Extract(:, :, 2), originalRAW*originalCOL, 1); 
           C_Extract = reshape(I_Extract(:, :, 3), originalRAW*originalCOL, 1); 
        end
    end
    if feaExtra == 2 % extract image HSV features
        I_Extract = rgb2hsv(I);
        if superpix == 0
           A_Extract = reshape(I_Extract(:, :, 1), originalRAW*originalCOL, 1); 
           B_Extract = reshape(I_Extract(:, :, 2), originalRAW*originalCOL, 1); 
           C_Extract = reshape(I_Extract(:, :, 3), originalRAW*originalCOL, 1); 
        end
    end
    % extract image RGB features
    if feaExtra == 3 % extract image RGB features
        I_Extract = im2double(I);
        if superpix == 0
        A = reshape(I(:, :, 1), originalRAW*originalCOL, 1);
        B = reshape(I(:, :, 2), originalRAW*originalCOL, 1);
        C = reshape(I(:, :, 3), originalRAW*originalCOL, 1);
        A_Extract = im2double(A); 
        B_Extract = im2double(B);
        C_Extract = im2double(C);
        end
    end

    if superpix == 1
        % use the superpixel SLIC to process the image
        s=7;
        errTh=10^-2;
        wDs=0.5^2;
        Label=SLIC(I,s,errTh,wDs);
        NB_Label = Label(originalRAW,originalCOL);
        point_line = cell(NB_Label,1);
        for i=1:originalRAW
            for j=1:originalCOL
                point_line{Label(i,j),1} = [point_line{Label(i,j),1}; I_Extract(i,j,1) I_Extract(i,j,2) I_Extract(i,j,3) i j];
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
        newRAW = [];
        newCOL = [];
    else
        % Simply extract the intrinsic properties of the image
        NB_Label = [];
        point_line = [];
        points = [A_Extract B_Extract C_Extract];
        newRAW = ceil(originalRAW*0.4);
        newCOL = ceil(originalCOL*0.4);
        [x,y,z]=size(points);
        takePointsRow = linspace(1,originalRAW,newRAW);
        for i=1:newRAW
            takePointsRow(i)=ceil(takePointsRow(i));
        end
        newPoints = [];
        count = 1;
        for i=1:x
            if mod(i,originalRAW) == takePointsRow(count)
                temp = points(i,:);
                newPoints = [newPoints ; temp];
                count = count + 1;
            end
            if count == newRAW
                count = 1;
            end
        end
        newX = (newRAW-1)*originalCOL;
        count = 1;
        newA = [];
        newB = [];
        newC = [];
        for i=1:newX
             temp = mod(i,newRAW-1);
             if temp == 0
                temp = temp + newRAW-1; 
             end 
             newA(temp,count) = newPoints(i,1);
             newB(temp,count) = newPoints(i,2);
             newC(temp,count) = newPoints(i,3);
             if temp == newRAW-1
                count = count + 1; 
             end
        end
        takePointsCol = linspace(1,originalCOL,newCOL);
        for i=1:newCOL
            takePointsCol(i)=ceil(takePointsCol(i));
        end
        countCol = 1;
        newA_ex = [];
        newB_ex = [];
        newC_ex = [];
        for i=1:originalCOL
            if i == takePointsCol(countCol)
                countCol = countCol + 1;
                newA_ex = [newA_ex , newA(:,i)];
                newB_ex = [newB_ex , newB(:,i)];
                newC_ex = [newC_ex , newC(:,i)];
            end
        end
        newI(:,:,1) = newA_ex;
        newI(:,:,2) = newB_ex;
        newI(:,:,3) = newC_ex;
        A_C = reshape(newI(:, :, 1), (newRAW-1)*newCOL , 1);    
        B_C = reshape(newI(:, :, 2), (newRAW-1)*newCOL , 1);
        C_C = reshape(newI(:, :, 3), (newRAW-1)*newCOL , 1);
        Lab_mean = [A_C B_C C_C]; 
    end
end