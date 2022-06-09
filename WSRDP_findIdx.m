function ids = WSRDP_findIdx(brushedData)
    ids = [];
    for i = 1:length(brushedData(:,1))
        for j = 1:length(X(:,1))
            if  brushedData(i,:) == X(j,:)
                ids = [ids; j];
            end
        end
    end
end