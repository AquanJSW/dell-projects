%�����ڶ���ֵ���㣻��Class�ĻҶȼ�Ȩ��ֵ
%k�Ǳ���ֵ���ֵ�class���±꣬arrayk�洢�˸���������ֵ
%m=(graymean(arrayk(1)))/cumulap(1),��k=1ʱ;
% =(��ip(i+1))/cumulap(2),i=arrayk(1)+1:arrayk(2),��k=2ʱ,etc. 
% =(��ip(i+1))/cumulap(k),i=arrayk(k-1)+1:255,��arrayk(k)=0ʱ
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
