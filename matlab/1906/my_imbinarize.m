function xoutput = my_imbinarize(xinput, T)
    T = T * 255;
    xoutput = xinput;
    xoutput(:,:) = 0;
    [a, b] = size(xoutput);
    for c = 1 : a
        for d = 1 : b
            if xinput(c, d) > T
                xoutput(c, d) = 255;
            end
        end
    end
end