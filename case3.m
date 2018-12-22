%% 构造数据集
data_len = 1000;
data = zeros(data_len,4);
data(:,1:3) = rand(data_len,3);
for i=1:data_len
    data(i,4) = ((data(i,1)-0.5)^2+(data(i,2)-0.5)^2+(data(i,3)-0.5)^2) <= 0.25^2;
end

gindex = find(data(:,4)==0);
rindex = find(data(:,4)==1);

figure;
scatter3(data(rindex,1),data(rindex,2),data(rindex,3),'filled','r');
hold on;
scatter3(data(gindex,1),data(gindex,2),data(gindex,3),'g');
title('训练集数据');

%% 训练
levels = [3,7,1];

[W,theta,record] = train(data,levels,0.5,1,@compute_error);

%% 测试
data_len = 1000;
data = zeros(data_len,4);
data(:,1:3) = rand(data_len,3);
for i=1:data_len
    data(i,4) = ((data(i,1)-0.5)^2+(data(i,2)-0.5)^2+(data(i,3)-0.5)^2) <= 0.25^2;
end

Y = predict(data(:,1:3),W,theta);

Y(Y>0.5)=1;
Y(Y<=0.5)=0;
T = data(:,4) - Y';
correct_index = find(T == 0);
green_index = find(data(:,4)==0);
error_index = find(T ~= 0);
red_index = find(data(:,4)==1);

figure;
scatter3(data(green_index,1),data(green_index,2),data(green_index,3),'g');
hold on
scatter3(data(red_index,1),data(red_index,2),data(red_index,3),'filled','r');
title('分类测试结果')
figure;
scatter3(data(correct_index,1),data(correct_index,2),data(correct_index,3),'g');
hold on
scatter3(data(error_index,1),data(error_index,2),data(error_index,3),'filled','r');
title('正确点和错误点')
figure;
plot(record);
title('历史错误率');
xlabel('次数');
ylabel('错误率');
