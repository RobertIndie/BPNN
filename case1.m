%% �������ݼ�
data_len = 1000;
data = zeros(data_len,3);
data(:,1:2) = rand(data_len,2);
for i=1:data_len
    data(i,3) = ((data(i,1)-0.5)^2+(data(i,2)-0.5)^2) <= 0.25^2;
end

gindex = find(data(:,3)==0);
rindex = find(data(:,3)==1);

figure;
scatter(data(rindex,1),data(rindex,2),'filled','r');
hold on;
scatter(data(gindex,1),data(gindex,2),'g');
title('ѵ��������');

%% ѵ��
levels = [2,7,1];

[W,theta,record] = BP_tranning(data,levels,0.5,1,@compute_error);

%% ����
data_len = 1000;
data = zeros(data_len,3);
data(:,1:2) = rand(data_len,2);
for i=1:data_len
    data(i,3) = ((data(i,1)-0.5)^2+(data(i,2)-0.5)^2) <= 0.25^2;
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
title('������Խ��')
figure;
scatter(data(correct_index,1),data(correct_index,2),'g');
hold on
scatter(data(error_index,1),data(error_index,2),'filled','r');
title('��ȷ��ʹ����')
figure;
plot(record);
title('��ʷ������');
xlabel('����');
ylabel('������');
