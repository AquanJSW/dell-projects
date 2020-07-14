%归一化像素数分布
function p = desityp(ximage)
	p = zeros(1, 256);
    [a, b] = size(ximage);
    for c = 1 : a
        for d = 1 : b
            p(ximage(c, d) + 1) = p(ximage(c, d) + 1) + 1;
        end
    end
    p = p / (a * b);   
end