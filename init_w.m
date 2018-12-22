function [ W ] = init_w( levels )
% 初始化连接权值矩阵，返回每一层之间的连接权值矩阵
    n_level = numel(levels) - 1;
    W = cell(n_level,1);
    for i=1:n_level
       W{i} = rand(levels(i+1),levels(i));
    end
end