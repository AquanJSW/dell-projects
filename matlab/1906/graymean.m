%0~255��k֮ǰ�ĻҶȼ�Ȩ��ֵ�������ڵ���ֵ�������䷽��Ŀ��ټ���
function m = graymean(k)
    global p;
    m = 0;
    for a = 0 : k
        m = m + a * p(a + 1);
    end
end