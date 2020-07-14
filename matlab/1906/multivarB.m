%用于多阈值分割的类间方差的计算
%N为阈值数目，限制N=2，这样var就是二维类间方差矩阵
function sigma_b_powered = multivarB(N)
	global mG;
    global arrayk;
    if N == 2
        sigma_b_powered = zeros(254, 254);
        for a = 1 : 253
            for b = (a + 1) : 254
                arrayk = [a, b, 0];
                for k = 1 : 3
                    sigma_b_powered(a, b) = sigma_b_powered(a, b) + cumulap(k) * (classgraymean(k) - mG).^2;
                end
            end
        end
    else
        error('N must be 2');
    end 
end