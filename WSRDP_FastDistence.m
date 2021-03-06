function  [rho, delta, ordrho, nneigh] = WSRDP_FastDistence(points_c, dc)
% 
K = 6;
% 
[idx_cell,d_cell] = rangesearch(points_c,points_c,dc);

ND=length(points_c(:,1));
for i=1:ND
  rho(i)=0.;
  delta(i)=0.;
  nneigh(i)=0.;
end
for i=1:ND
    rho(i) = length(cell2mat(idx_cell(i)))-1;
end
count_density_peak = 0;
count_outliers = 0;
local_density_peak_first = [];
%
for i=1:ND
    if rho(i) == 0
        count_outliers = count_outliers + 1;
        outliers(count_outliers) = i;
        continue;
    end
    temp_d = cell2mat(d_cell(i))';
    temp_id = cell2mat(idx_cell(i))';
    temp_rho = zeros(length(temp_id),1);  
    for j=1:length(temp_id)
        temp_rho(j) = rho(temp_id(j));
    end
    temp_id_d_rho = [temp_id temp_d temp_rho];
    temp_id_d_rho_sortrho = sortrows(temp_id_d_rho,-3);
    if temp_id_d_rho_sortrho(1,1) == i  || temp_id_d_rho_sortrho(1,3) == rho(i)
        count_density_peak = count_density_peak + 1;
        local_density_peak(count_density_peak) = i;
        local_density_peak_first = [local_density_peak_first;points_c(i,:)];
        continue;
    end
    for j=1:length(temp_id)
        if(i == temp_id_d_rho_sortrho(j,1))
            temp_marix = temp_id_d_rho_sortrho(1:j-1,:);
            temp_marix_sortdelta = sortrows(temp_marix,-2);
            flag = 0;
            for w= j:length(temp_id_d_rho_sortrho(:,1))
                if temp_id_d_rho_sortrho(w,2) ~= 0
                   if temp_id_d_rho_sortrho(w,1)<i && temp_id_d_rho_sortrho(w,3) == temp_id_d_rho_sortrho(j,3)...
                                    && temp_id_d_rho_sortrho(j-1,2) ~= 0
                      delta(i) = temp_id_d_rho_sortrho(w,2);
                      nneigh(i) = temp_id_d_rho_sortrho(w,1);
                      flag = 1;
                    end
                    break;
                end
            end
            if flag == 1
               break;
            end
            length_marix = length(temp_marix_sortdelta(:,1));
            for o= length_marix:-1:1
               if temp_marix_sortdelta(1,2) == temp_marix_sortdelta(length_marix,2)
                  delta(i) = temp_marix_sortdelta(1,2);
                  nneigh(i) = temp_marix_sortdelta(1,1);
               end
               if temp_marix_sortdelta(o,2) == temp_marix_sortdelta(length_marix,2)
                  continue;
               end
               delta(i) = temp_marix_sortdelta(o+1,2);
               nneigh(i) = temp_marix_sortdelta(o+1,1);
               break;
            end
            break;
        end
    end
end
dc_plus = dc;
count_density_peak_plus = 0;
local_density_peak_plus = [];
if count_density_peak > 1
    for i=2:K
        dc_plus = dc_plus*i;
        i
        length(local_density_peak)
        for j=1:length(local_density_peak)
            local_density_peak(j)
            [idx_cell_plus,d_cell_plus] = rangesearch(points_c(local_density_peak(j),:),points_c,dc_plus);
            temp_d = [];
            temp_id = [];
            for c=1:ND
                if cell2mat(idx_cell_plus(c)) == 1
                    temp_id = [temp_id ; c];
                    temp_d = [temp_d ; cell2mat(d_cell_plus(c))];
                end 
            end
            temp_rho = zeros(length(temp_id),1);
            for k=1:length(temp_id)
                temp_rho(k) = rho(temp_id(k));
            end
            temp_id_d_rho = [temp_id temp_d temp_rho];
            temp_id_d_rho_sortrho = sortrows(temp_id_d_rho,-3);
            if temp_id_d_rho_sortrho(1,1) == local_density_peak(j) || temp_id_d_rho_sortrho(1,3) == rho(local_density_peak(j))
                count_density_peak_plus = count_density_peak_plus + 1;
                local_density_peak_plus(count_density_peak_plus) = local_density_peak(j);
                continue;
            end
            for k=1:length(temp_id)
                if(local_density_peak(j) == temp_id_d_rho_sortrho(k,1))
                    temp_marix = temp_id_d_rho_sortrho(1:k-1,:);
                    temp_marix_sortdelta = sortrows(temp_marix,-2);
                    flag = 0;
                    for w= k:length(temp_id_d_rho_sortrho(:,1))
                        if temp_id_d_rho_sortrho(w,2) ~= 0
                            if temp_id_d_rho_sortrho(w,1)<local_density_peak(j) && temp_id_d_rho_sortrho(w,3) == temp_id_d_rho_sortrho(k,3)...
                                    && temp_id_d_rho_sortrho(k-1,2) ~= 0
                                delta(local_density_peak(j)) = temp_id_d_rho_sortrho(w,2);
                                nneigh(local_density_peak(j)) = temp_id_d_rho_sortrho(w,1);
                                flag = 1;
                            end
                            break;
                        end
                    end
                    if flag == 1
                       break;
                    end
                    length_marix = length(temp_marix_sortdelta(:,1));
                    for o= length_marix:-1:1
                        if temp_marix_sortdelta(1,2) == temp_marix_sortdelta(length_marix,2)
                            delta(local_density_peak(j)) = temp_marix_sortdelta(1,2);
                            nneigh(local_density_peak(j)) = temp_marix_sortdelta(1,1);
                        end
                        if temp_marix_sortdelta(o,2) == temp_marix_sortdelta(length_marix,2)
                            continue;
                        end
                        delta(local_density_peak(j)) = temp_marix_sortdelta(o+1,2);
                        nneigh(local_density_peak(j)) = temp_marix_sortdelta(o+1,1);
                        break;
                    end
                    break;
                end              
            end
        end
       if length(local_density_peak_plus) == 1
            break;
       end
       count_local_density_peak_plus = 1;
       flag = 0;
       for u = 2:length(local_density_peak_plus)
           if rho(local_density_peak_plus(1)) == rho(local_density_peak_plus(u))
               count_local_density_peak_plus = count_local_density_peak_plus + 1;
           else
               break;
           end
           if count_local_density_peak_plus == length(local_density_peak_plus)
               flag = 1;
           end
       end
       if flag == 1
           break;
       end
       if length(local_density_peak) == length(local_density_peak_plus)
           break;
       end
       local_density_peak = [];
       local_density_peak = local_density_peak_plus;
       local_density_peak_plus = [];
       count_density_peak_plus = 0;
    end
end
length(local_density_peak_plus)
if length(local_density_peak_plus) > 1
    fprintf('Note: the length of local_density_peak_plus exceeds 1, its value is %d\n',length(local_density_peak_plus));
    delta(local_density_peak_plus(1)) = max(delta(:));
    for i=2:length(local_density_peak_plus)
        delta(local_density_peak_plus(i)) = pdist2(points_c(local_density_peak_plus(i)),points_c(local_density_peak_plus(1)));
        nneigh(local_density_peak_plus(i)) = local_density_peak_plus(1);
    end
end
if length(local_density_peak_plus) == 1
    delta(local_density_peak_plus(1)) = max(delta(:));
end

dc_plus = dc;
count_outliers_plus = 0;
outliers_plus = [];
if count_outliers > 0
    for i=2:ND
        dc_plus = dc_plus*i;        
        for j=1:length(outliers)
            [idx_cell_plus,d_cell_plus] = rangesearch(points_c(outliers(j),:),points_c,dc_plus);
            temp_d = [];
            temp_id = [];
            for c=1:ND
                if cell2mat(idx_cell_plus(c)) == 1
                    temp_id = [temp_id ; c];
                    temp_d = [temp_d ; cell2mat(d_cell_plus(c))];
                end  
            end
            temp_rho = zeros(length(temp_id),1);
            for k=1:length(temp_id)
                temp_rho(k) = rho(temp_id(k));
            end
            temp_id_d_rho = [temp_id temp_d temp_rho];
            temp_id_d_rho_sortrho = sortrows(temp_id_d_rho,-3);
            if temp_id_d_rho_sortrho(1,1) == outliers(j) || temp_id_d_rho_sortrho(1,3) == rho(outliers(j))
                count_outliers_plus = count_outliers_plus + 1;
                outliers_plus(count_outliers_plus) = outliers(j);
                continue;
            end
            for k=1:length(temp_id)
                if(outliers(j) == temp_id_d_rho_sortrho(k,1))
                    temp_marix = temp_id_d_rho_sortrho(1:k-1,:);
                    temp_marix_sortdelta = sortrows(temp_marix,-2);
                    flag = 0;
                    for w= k:length(temp_id_d_rho_sortrho(:,1))
                        if temp_id_d_rho_sortrho(w,2) ~= 0
                            if temp_id_d_rho_sortrho(w,1)<outliers(j) && temp_id_d_rho_sortrho(w,3) == temp_id_d_rho_sortrho(k,3)...
                                    && temp_id_d_rho_sortrho(k-1,2) ~= 0
                                delta(outliers(j)) = temp_id_d_rho_sortrho(w,2);
                                nneigh(outliers(j)) = temp_id_d_rho_sortrho(w,1);
                                flag = 1;
                            end
                            break;
                        end
                    end
                    if flag == 1
                       break;
                    end
                    length_marix = length(temp_marix_sortdelta(:,1));
                    for o= length_marix:-1:1
                        if temp_marix_sortdelta(1,2) == temp_marix_sortdelta(length_marix,2)
                            delta(outliers(j)) = temp_marix_sortdelta(1,2);
                            nneigh(outliers(j)) = temp_marix_sortdelta(1,1);
                        end
                        if temp_marix_sortdelta(o,2) == temp_marix_sortdelta(length_marix,2)
                            continue;
                        end
                        delta(outliers(j)) = temp_marix_sortdelta(o+1,2);
                        nneigh(outliers(j)) = temp_marix_sortdelta(o+1,1);
                        break;
                    end
                    break;
                end              
            end 
        end
        
        if length(outliers) == 0
            break;
        end
        outliers = [];
        outliers = outliers_plus;
        outliers_plus = [];
        count_outliers_plus = 0;
        
    end
end

% get the final rho and delta
% If rho and delta have an assignment error, the reason is the wrong rho and delta are selected.
[rho,PS] = mapminmax(rho,0,1);
[delta,PS] = mapminmax(delta,0,1);
[rho_sorted,ordrho]=sort(rho,'descend');

end