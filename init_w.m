function [ W ] = init_w( levels )
% ��ʼ������Ȩֵ���󣬷���ÿһ��֮�������Ȩֵ����
    n_level = numel(levels) - 1;
    W = cell(n_level,1);
    for i=1:n_level
       W{i} = rand(levels(i+1),levels(i));
    end
end