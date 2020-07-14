function [T, eta] = my_graythresh(ximage)
    T = 0;
%归一化像素数序列p
    global p;
    p = desityp(ximage);
%全局灰度均值
    global mG;
    mG = graymean(255);
%寻找阈值T
    var = zeros(1, 256);%存放类间方差
    flag = 1;%最大类间方差的下标
    counter = 0;%最大类间方差的个数   
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
%衡量T好坏的无量纲参数eta
    varG = 0;%全局方差
    for k = 0 : 255
        varG = varG + p(k + 1) * (k - mG).^2;
    end
    
    if varG == 0
        eta = 0;
    else
        eta = var(flag) / varG;
    end
end