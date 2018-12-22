%% BP神经网络的输入输出数据
% clear all
% clc
% x=rand(1,1500);
% y=rand(1,1500);
% p=[x;y]';
% z=x.^2-y.^2+3;
% save x x
% save y y
% save p p
% save z z 
%%
clear all
clc
%% 训练数据预测数据提取及归一化
% 下载输入输出数据
% load x y z p
%x=rand(1,1500);
x=linspace(0,10*2*pi,1500);
%y=rand(1,1500);
y=sin(x);
p=[x;y]';
z=sin(x);
% 从1到1500间随机排序
k=rand(1,1500);
[m,n]=sort(k);
% 找出训练数据和预测数据
input_train=x(1:1400)';
output_train=z(1:1400);
input_test=x(1401:1500)';
output_test=z(1401:1500);
% 训练样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
%% BP网络训练
% 初始化网络结构
net=newff(inputn,outputn,5);

net.trainParam.epochs=500;
net.trainParam.lr=0.05;
net.trainParam.goal=0.0004;
% 网络训练
net=train(net,inputn,outputn);
%% BP网络预测
% 预测数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
% 网络预测输出
an=sim(net,inputn_test);
% 网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);
%% 结果分析
figure(1)
plot(BPoutput,':og');
hold on
plot(output_test,'-*');
ylabel('仿真输出','fontsize',12);
xlabel('样本','fontsize',12);
% 预测误差
error=BPoutput-output_test;
figure(2)
subplot(2,1,1)
plot(error,'-*');
ylabel('绝对误差','fontsize',12);
xlabel('样本','fontsize',12);

subplot(2,1,2)
plot((output_test-BPoutput)./BPoutput*100,'-*');
ylabel('误差百分比','fontsize',12)
xlabel('样本','fontsize',12)

errorsum=sum(abs(error))
