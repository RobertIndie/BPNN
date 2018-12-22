%% 构造数据集
data_len = 1000;
data = zeros(data_len,2);
data(:,1) = linspace(0,2*pi,data_len);%rand(data_len,1)*2*pi;
data(:,2) = sin(data(:,1))/2+0.5;
scatter(data(:,1),data(:,2))
data(:,2) = data(:,2);

%% 训练
levels = [1,7,1];

[W,theta,record] = BP_tranning(data,levels,1,19);

%% 测试

output = BP_predict2(data(:,1),W,theta);
hold on;
scatter(data(:,1),output)
                   