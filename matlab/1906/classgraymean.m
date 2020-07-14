%适用于多阈值计算；单Class的灰度加权均值
%k是被阈值划分的class的下标，arrayk存储了各个待测阈值
%m=(graymean(arrayk(1)))/cumulap(1),当k=1时;
% =(Σip(i+1))/cumulap(2),i=arrayk(1)+1:arrayk(2),当k=2时,etc. 
% =(Σip(i+1))/cumulap(k),i=arrayk(k-1)+1:255,当arrayk(k)=0时
function m = classgraymean(k)
    global p;
    m = 0;
    global arrayk;
    if cumulap(k) == 0
        return;
    else
        if k == 1
            m = (graymean(arrayk(1))) / cumulap(1);
        elseif (k > 1) && (arrayk(k) > 0)
            for a = arrayk(k - 1) + 1 : arrayk(k)
                m = m + a * p(a + 1);
            end
            m = m / cumulap(k);
        elseif arrayk(k) == 0
            for a = arrayk(k - 1) + 1 : 255
                m = m + a * p(a + 1);
            end
            m = m / cumulap(k);
        end
    end
end
