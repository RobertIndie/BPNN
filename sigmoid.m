function [ out ] = sigmoid( in )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here

    out = 1./(1+exp(-in));
    
end