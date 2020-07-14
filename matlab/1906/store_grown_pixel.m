%存储生长像素坐标的两个堆栈：grown_pixels_x, grown_pixels_y;
%这里的seed_num表示原始种子编号
function store_grown_pixel(x, y, seed_num)
    global grown_pixels_x;
    global grown_pixels_y;
    global region_size;
    global temp_image;
    
    temp_image(x, y) = 255;%将缓存图像上的该种子点置255
    
    grown_pixels_x(region_size(seed_num) + 1, seed_num) = x;
    grown_pixels_y(region_size(seed_num) + 1, seed_num) = y;
    region_size(seed_num) = region_size(seed_num) + 1;
end