%生成部分归一化像素数累加和 P1(k)
function P = P1(k)
    global p;
    P = 0;
    for a = 1 : (k + 1)
        P = P + p(a);
    end
end