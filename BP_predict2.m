function [ output] = BP_predict2( input, W, theta)
%BP_ Summary of this function goes here
%   Detailed explanation goes here

    f = @sigmoid;
    n_w = numel(W);
    y = input';

    for i=1:n_w
        net = bsxfun(@plus,W{i}*y,theta{i});
        y = f(net);
    end
    output = y;
end
