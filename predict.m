function [ output,Y] = predict( input, W, theta)
% 前向传播
% input: 输入向量
% W: 连接权值矩阵元胞数组
% theta: 阈值矩阵
% output: 输出向量
% Y: 各层的输出
    f = @sigmoid;
    n_w = numel(W);
    Y = cell(n_w+1, 1);
    y = input';
    Y{1}=input';
    for i=1:n_w
        V = W{i}*y+theta{i};
        y = f(V);
        Y{i+1} = y;
    end
    output = y;
end
