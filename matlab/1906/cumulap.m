%���ɲ��ֹ�һ���������ۼӺ�
%k�Ǳ���ֵ���ֵ�class���±꣬arrayk�洢�˸���������ֵ
function P = cumulap(k)
    global p;
    P = 0;
    global arrayk;
    if k == 1
        for a = 0 : arrayk(1)
            P = P + p(a + 1);
        end
    elseif arrayk(k) == 0
        for a = (arrayk(k - 1) + 1) : 255
            P = P + p(a + 1);
        end
    else
        for a = (arrayk(k - 1) + 1) : arrayk(k)
            P = P + p(a + 1);
        end
    end
end