function [ theta ] = init_theta( levels )
% ≥ı ºªØ„–÷µæÿ’Û
    n_level = numel(levels) - 1;
    theta = cell(n_level, 1);
    for i=1:n_level
       theta{i} = rands(levels(i+1),1); 
    end
end
