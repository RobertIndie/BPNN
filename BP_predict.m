function [ output, Y ] = BP_predict( input, W, theta)
%BP_ Summary of this function goes here
%   Detailed explanation goes here
    f = @sigmoid;
    n_w = numel(W);
    Y = cell(n_w+1, 1);
    Y{1} = input';

    for i=1:n_w
        net = W{i}*Y{i} + theta{i};
        Y{i+1} = f(net);
    end
    output = Y{end};
end
