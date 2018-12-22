%% Build a tranning set of three class
c_1 = [0 0];
c_2 = [0 1];
c_3 = [1 0];
c_4 = [1 1];

n_L1 = 20; % number of label 1
n_L2 = 20; % number of label 2
n_L3 = 20; % number of label 3

A = zeros(n_L1, 4);
A(:,3) = 0; A(:,4) = 0;
for i=1:n_L1
    A(i,1:2) = c_1 + rand(1,2)/2;
end

B = zeros(n_L2, 4);
B(:,3) = 0; B(:,4) = 1;
for i=1:n_L2
    B(i,1:2) = c_2 + rand(1,2)/2;
end
C = zeros(n_L3, 4);
C(:,3) = 1; C(:,4) = 0;
for i=1:n_L3
    C(i,1:2) = c_3 + rand(1,2)/2;
end
D = zeros(n_L3, 4);
D(:,3) = 1; D(:,4) = 0;
for i=1:n_L3
    D(i,1:2) = c_4 + rand(1,2)/2;
end
scatter(A(:,1), A(:,2),[],'r');
hold on
scatter(B(:,1), B(:,2),[],'g');
hold on
scatter(C(:,1), C(:,2),[],'k');
hold on
scatter(D(:,1), D(:,2),[],'k');
X = [A;B;C;D];

%% Trainning the BP network
dbstop if error
n_label = 2;
% create the weight matrix
n_input = size(X,2) - n_label;% number of feature of each data, here is 2, only x-axis and y-axis
levels = [n_input 7 n_label];

step = 0.1;

[W,theta,record] = BP_tranning(X, levels, step,1);
save three_class W theta

%% show the result
input = X(:,1:n_input);
label = X(:,n_input+1:end);
y = BP_predict2(input, W, theta);
Y = y';
Y(Y>0.5) = 1;
Y(Y<=0.5) = 0;
T = sum(label - Y, 2);
error_index = find(T ~= 0);
figure;
scatter(X(:,1), X(:,2), [], 'g');
hold on
scatter(X(error_index,1), X(error_index,2), [], 'r');
