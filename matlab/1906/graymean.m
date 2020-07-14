%0~255中k之前的灰度加权均值，仅用于单阈值情况的类间方差的快速计算
function m = graymean(k)
    global p;
    m = 0;
    for a = 0 : k
        m = m + a * p(a + 1);
    end
end