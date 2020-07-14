%部分灰度均值
function m = graymean(k)
    global p;
    m = 0;
    for a = 0 : k
        m = m + a * p(a + 1);
    end
end