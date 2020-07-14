function [T, eta] = my_graythresh(ximage)
    T = 0;
%��һ������������p
    global p;
    p = desityp(ximage);
%ȫ�ֻҶȾ�ֵ
    global mG;
    mG = graymean(255);
%Ѱ����ֵT
    var = zeros(1, 256);%�����䷽��
    flag = 1;%�����䷽����±�
    counter = 0;%�����䷽��ĸ���   
    for k = 0 : 255
        var(k + 1) = varB(k);
    end 
    for k = 1 : 255
        if var(k + 1) > var(flag)
            flag = k + 1;
        end
    end
    for k = 0 : 255
        if var(k + 1) == var(flag)
            T = T + k;
            counter = counter + 1;
        end
    end
    T = T / counter / 255;
%����T�û��������ٲ���eta
    varG = 0;%ȫ�ַ���
    for k = 0 : 255
        varG = varG + p(k + 1) * (k - mG).^2;
    end
    
    if varG == 0
        eta = 0;
    else
        eta = var(flag) / varG;
    end
end