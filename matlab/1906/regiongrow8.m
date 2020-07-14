%8邻域生长并返回离散图像
%T=0~1,作为停止生长的阈值，越大表示限制越少，0表示必须与种子灰度值相同，1表示
%无限制（此时无意义）
function outimage = regiongrow8(inimage, T)
    %种子坐标，均为n*1向量
    global seedx;
    global seedy;
    
    global region_num;
    [region_num, ~] = size(seedx);%原始种子数目(区域数目)
    
    [lenx_image, leny_image] = size(image);
    pixels_image = lenx_image * leny_image;%图像像素数
    
    %存储生长像素的坐标
    global grown_pixels_x;
    global grown_pixels_y;
    grown_pixels_x = zeros(pixels_image, region_num);
    grown_pixels_y = zeros(pixels_image, region_num);
    
    %存储已生长的区域中像素数
    global region_size;
    region_size = zeros(1, region_num);
    
    %二值缓存图像
    global temp_image;
    temp_image = inimage;
    temp_image = 0;
    
    %复制元种子坐标
    for a = 1 : region_num
        store_grown_pixel([seedx(a), seedy(a)], a);
    end
    

    
    %生长
    grow_start(inimage, T);
    
    outimage = temp_image;
end