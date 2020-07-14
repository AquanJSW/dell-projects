%多阈值分割算法
%N为阈值数目，这里只编写了N=2时的算法，因为N>2时要用到基于搜索的优化算法，我们
%目的在于了解多阈值分割算法，而该算法对于任意N都是通用的
function [thresholds_array, eta] = my_multithresh(ximage, N)
    %归一化像素数分布
    global p;
    p = desityp(ximage);
    
    %全局灰度均值
    global mG;
    mG = graymean(255);
    
    if N > 2
        error('Please use MULTITHRESH');
    elseif N == 2
        sigma_b_powered = multivarB(N);%类间方差矩阵255*255
        counter_max_sigma_b_powered = 0;%最大类间方差个数
        
        %寻找最大类间方差下标
        flag_max_sigma_b_powered = find(max(max(sigma_b_powered)) == sigma_b_powered);
        
        %寻找最大类间方差个数并累加下标(灰度)
        thresholds_array = zeros(1,2);
        for a = 1 : (255 - N)
            for b = (a + 1) : 254
                if sigma_b_powered(a, b) == sigma_b_powered(flag_max_sigma_b_powered)
                     thresholds_array= thresholds_array + [a, b];
                    counter_max_sigma_b_powered = counter_max_sigma_b_powered + 1;
                end
            end
        end
        
        %得出阈值向量
        thresholds_array = thresholds_array / counter_max_sigma_b_powered;
        
        %得出可分离性参数
        sigma_g_powered = 0;%全局方差
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