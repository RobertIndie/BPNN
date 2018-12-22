function [ W,theta,record ] = train( X, levels, step, min_err, computeErrorFunc )
% 训练
% X: 训练集
% levels: 神经网络结构
% step: 趋近学习步长
% min_err: 期望错误率
% computeErrorFunc: 错误率计算函数
    r_i = 0;record = zeros(1,10000);% 历史错误率
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
           % 从最后一层往前更新W
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
        record(r_i) = error; % 记录历史错误率
        if error < min_err
            break;
        end
    end
    record=record(1,1:r_i);
end
