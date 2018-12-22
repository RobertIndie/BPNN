%% BP������������������
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
%% ѵ������Ԥ��������ȡ����һ��
% ���������������
% load x y z p
%x=rand(1,1500);
x=linspace(0,10*2*pi,1500);
%y=rand(1,1500);
y=sin(x);
p=[x;y]';
z=sin(x);
% ��1��1500���������
k=rand(1,1500);
[m,n]=sort(k);
% �ҳ�ѵ�����ݺ�Ԥ������
input_train=x(1:1400)';
output_train=z(1:1400);
input_test=x(1401:1500)';
output_test=z(1401:1500);
% ѵ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
%% BP����ѵ��
% ��ʼ������ṹ
net=newff(inputn,outputn,5);

net.trainParam.epochs=500;
net.trainParam.lr=0.05;
net.trainParam.goal=0.0004;
% ����ѵ��
net=train(net,inputn,outputn);
%% BP����Ԥ��
% Ԥ�����ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
% ����Ԥ�����
an=sim(net,inputn_test);
% �����������һ��
BPoutput=mapminmax('reverse',an,outputps);
%% �������
figure(1)
plot(BPoutput,':og');
hold on
plot(output_test,'-*');
ylabel('�������','fontsize',12);
xlabel('����','fontsize',12);
% Ԥ�����
error=BPoutput-output_test;
figure(2)
subplot(2,1,1)
plot(error,'-*');
ylabel('�������','fontsize',12);
xlabel('����','fontsize',12);

subplot(2,1,2)
plot((output_test-BPoutput)./BPoutput*100,'-*');
ylabel('���ٷֱ�','fontsize',12)
xlabel('����','fontsize',12)

errorsum=sum(abs(error))
