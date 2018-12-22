function [ output,Y] = predict( input, W, theta)
% ǰ�򴫲�
% input: ��������
% W: ����Ȩֵ����Ԫ������
% theta: ��ֵ����
% output: �������
% Y: ��������
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
