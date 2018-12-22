%% 构造数据集
data_len = 1000;
data = zeros(data_len,3);
data(:,1:2) = rand(data_len,2);
for i=1:data_len
    data(i,3) = ((data(i,1)-0.5)^2-(data(i,2)-0.5)^2) <= 0.25^2;
end

gindex = find(data(:,3)==0);
rindex = find(data(:,3)==1);

figure;
scatter(data(rindex,1),data(rindex,2),'filled','r');
hold on;
scatter(data(gindex,1),data(gindex,2),'g');
title('训练集数据');

%% 训练
levels = [2,7,1];

[W,theta,record] = BP_tranning(data,levels,0.5,1,@compute_error);

%% 测试
data_len = 1000;
data = zeros(data_len,3);
data(:,1:2) = rand(data_len,2);
for i=1:data_len
    data(i,3) = ((data(i,1)-0.5)^2-(data(i,2)-0.5)^2) <= 0.25^2;
end

Y = predict(data(:,1:2),W,theta);

Y(Y>0.5)=1;
Y(Y<=0.5)=0;
T = data(:,3) - Y';
correct_index = find(T == 0);
green_index = find(data(:,3)==0);
error_index = find(T ~= 0);
red_index = find(data(:,3)==1);

figure;
scatter(data(green_index,1),data(green_index,2),'g');
hold on
scatter(data(red_index,1),data(red_index,2),'filled','r');
title('分类测试结果')
figure;
scatter(data(correct_index,1),data(correct_index,2),'g');
hold on
scatter(data(error_index,1),data(error_index,2),'filled','r');
title('正确点和错误点')
figure;
plot(record);
title('历史错误率');
xlabel('次数');
ylabel('错误率');

save case2 W theta