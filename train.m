function [ W,theta,record ] = train( X, levels, step, min_err, computeErrorFunc )
% ѵ��
% X: ѵ����
% levels: ������ṹ
% step: ����ѧϰ����
% min_err: ����������
% computeErrorFunc: �����ʼ��㺯��
    r_i = 0;record = zeros(1,10000);% ��ʷ������
    n_levels = size(levels, 2) - 1;
    n_input = levels(1);
    W = init_w(levels);
    theta = init_theta(levels);
    [n_data,col] = size(X);
    train_data = X(:,1:n_input); 
    label = X(:,n_input+1:end)';
    f = @sigmoid;
    while true
        for k=1:n_data
           [output, Y] =  predict(train_data(k,:), W, theta);
           delta = label(:,k) - output;
           % �����һ����ǰ����W
           for l=n_levels:-1:1
              net = W{l}*Y{l} + theta{l};
              if l == n_levels
                  S = diag(f(net).*(1-f(net)))*delta;
              else
                  S = diag((f(net).*(1-f(net))))*W{l+1}'*S;
              end
              new_W{l} = W{l} + step*S*Y{l}';
              new_theta{l} = theta{l} + step*S;
           end 
           W = new_W;
           theta = new_theta;
        end 

        y = predict(train_data, W, theta);
        error = computeErrorFunc(y, label)
        r_i = r_i + 1;
        record(r_i) = error; % ��¼��ʷ������
        if error < min_err
            break;
        end
    end
    record=record(1,1:r_i);
end
