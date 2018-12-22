%% 构造数据集
data_len = 1000;
data = zeros(data_len,3);
data(:,1:2) = rand(data_len,2);
for i=1:data_len
    data(i,3) = ((data(i,1)-0.5)^2-(data(i,2)-0.5)^2) <= 0.25^2;
end

a_len = size(data(:,3)==1,1);
show_data_a = zeros(a_len,2);
i_a=1;
show_data_b = zeros(data_len-a_len,1);
i_b=1;
for i=1:data_len
    if data(i,3)==1
        show_data_a(i_a,1:2)=data(i,1:2);
        i_a = i_a + 1;
    else
        show_data_b(i_b,1:2)=data(i,1:2);
        i_b = i_b + 1;
    end
end

scatter(show_data_a(:,1),show_data_a(:,2),'r');
hold on;
scatter(show_data_b(:,1),show_data_b(:,2),'g');

%% 训练
levels = [2,7,1];

[W,theta,record] = BP_tranning(data,levels,1,2);

%% 测试
data_len = 100;
data = zeros(data_len,3);
data(:,1:2) = rand(data_len,2);
for i=1:data_len
    data(i,3) = ((data(i,1)-0.5)^2-(data(i,2)-0.5)^2) <= 0.25^2;
end

Y = BP_predict2(data(:,1:2),W,theta);

Y(Y>0.5)=1;
Y(Y<=0.5)=0;
T = data(:,3) - Y';
correct_index = find(T == 0);
error_index = find(T ~= 0);

figure;
scatter(data(correct_index,1),data(correct_index,2),'g');
hold on
scatter(data(error_index,1),data(error_index,2),'r');
