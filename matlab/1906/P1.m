%���ɲ��ֹ�һ���������ۼӺ� P1(k)
function P = P1(k)
    global p;
    P = 0;
    for a = 1 : (k + 1)
        P = P + p(a);
    end
end