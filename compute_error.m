function [ error ] = compute_error(output,target)
%COMPUTE_ERROR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    output(output>0.5)=1;
    output(output<=0.5)=0;
    delta = abs(output - target);
    error = sum(sum(delta))/size(delta,2)*100;
end

