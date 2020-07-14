%����ֵ�ָ��㷨
%NΪ��ֵ��Ŀ������ֻ��д��N=2ʱ���㷨����ΪN>2ʱҪ�õ������������Ż��㷨������
%Ŀ�������˽����ֵ�ָ��㷨�������㷨��������N����ͨ�õ�
function [thresholds_array, eta] = my_multithresh(ximage, N)
    %��һ���������ֲ�
    global p;
    p = desityp(ximage);
    
    %ȫ�ֻҶȾ�ֵ
    global mG;
    mG = graymean(255);
    
    if N > 2
        error('Please use MULTITHRESH');
    elseif N == 2
        sigma_b_powered = multivarB(N);%��䷽�����255*255
        counter_max_sigma_b_powered = 0;%�����䷽�����
        
        %Ѱ�������䷽���±�
        flag_max_sigma_b_powered = find(max(max(sigma_b_powered)) == sigma_b_powered);
        
        %Ѱ�������䷽��������ۼ��±�(�Ҷ�)
        thresholds_array = zeros(1,2);
        for a = 1 : (255 - N)
            for b = (a + 1) : 254
                if sigma_b_powered(a, b) == sigma_b_powered(flag_max_sigma_b_powered)
                     thresholds_array= thresholds_array + [a, b];
                    counter_max_sigma_b_powered = counter_max_sigma_b_powered + 1;
                end
            end
        end
        
        %�ó���ֵ����
        thresholds_array = thresholds_array / counter_max_sigma_b_powered;
        
        %�ó��ɷ����Բ���
        sigma_g_powered = 0;%ȫ�ַ���
        for k = 0 : 255
            sigma_g_powered = sigma_g_powered + p(k + 1) * (k - mG).^2;
        end
        if sigma_g_powered == 0
            eta = 0;
        else
            eta = sigma_b_powered(flag_max_sigma_b_powered) / sigma_g_powered;
        end
    end
end