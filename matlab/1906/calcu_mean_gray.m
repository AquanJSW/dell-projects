%计算已有种子的灰度均值
%这里的seed_num表示原始种子编号
function mean_gray = calcu_mean_gray(inimage, seed_num)
    global grown_pixels_x;
    global grown_pixels_y;
    global region_size;
    
    mean_gray = 0;
    for a = 1 : region_size(seed_num)
        mean_gray = mean_gray + double(inimage(grown_pixels_x(a, seed_num), grown_pixels_y(a, seed_num)));
    end
    mean_gray = mean_gray / region_size(seed_num);
end