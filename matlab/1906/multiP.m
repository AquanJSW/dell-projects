%计算多阈值分割中部分归一化像素数和P
function P = multiP(k1, k2)
    global p;
    P = 0;
    if nargin == 1
        P = P1(k1);
    elseif nargin > 2
        P = NaN;
    elseif nargin == 2
        if k1 < k2
        	for a = k1 + 1 : k2 + 1
                P = P + p(a + 1)
            end
        end
    end
end