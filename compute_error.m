function [ error ] = compute_error(output,target)
%COMPUTE_ERROR 此处显示有关此函数的摘要
%   此处显示详细说明
    output(output>0.5)=1;
    output(output<=0.5)=0;
    delta = abs(output - target);
    error = sum(sum(delta))/size(delta,2)*100;
end

