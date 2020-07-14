%区域生长法分割图像
%T=0~1,作为停止生长的阈值，越大表示限制越少，0表示必须与种子灰度值相同，1表示
%无限制（此时无意义），默认情况0.5
%简单起见，结果使用二值显示
%注意，对inimage(a,b)用[x,y]=ginput，其中x对应b,y对应a
%inimage(grown_pixels_x(b, a) + c, grown_pixels_y(b, a) + d) 本式参与任何运算
%都会强行将结果转化在0~255之内，所以要用double
function outimage = regiongrowseg(inimage, T)
    %设置默认阈值
    if nargin == 1
        T = 0.5;
    end
    T = T * 255;
    
    %鼠标取若干个点作为种子(seed)，回车结束
    global seedy;
    global seedx;
    imshow(inimage);
    [seedy, seedx] = ginput;
    seedx = round(seedx);
    seedy = round(seedy);
    
    global region_num;
    [region_num, ~] = size(seedx);%原始种子数目(区域数目)
    
    [lenx_inimage, leny_inimage] = size(inimage);
    pixels_inimage = lenx_inimage * leny_inimage;%图像像素数
    
    %存储所有种子像素坐标的两个堆栈：grown_pixels_x, grown_pixels_y;
    global grown_pixels_x;
    global grown_pixels_y;
    grown_pixels_x = zeros(pixels_inimage, region_num);
    grown_pixels_y = zeros(pixels_inimage, region_num);
    
    %存储已生长的区域中像素数
    global region_size;
    region_size = zeros(1, region_num);
    
    %二值缓存图像
    global temp_image;
    temp_image = zeros(lenx_inimage, leny_inimage);
    
    %逐个种子生长
    for a = 1 : region_num
        store_grown_pixel(seedx(a), seedy(a), a);%元种子坐标入栈
        counter_new_seed = 1;
        while counter_new_seed
            maxb = region_size(a);
            minb = maxb - counter_new_seed + 1;
            counter_new_seed = 0;
            mean_gray = calcu_mean_gray(inimage, a);
            for b = minb : maxb
                for c = -1 : 1
                    for d = -1 : 1
                        if grown_pixels_x(b, a) + c > 0 ...
                        && grown_pixels_x(b, a) + c <= lenx_inimage ...
                        && grown_pixels_y(b, a) + d > 0 ...
                        && grown_pixels_y(b, a) + d <= leny_inimage ...
                        && temp_image(grown_pixels_x(b, a) + c, grown_pixels_y(b, a) + d) == 0 ...
                        && abs(double(inimage(grown_pixels_x(b, a) + c, grown_pixels_y(b, a) + d)) - mean_gray) <= T
                            store_grown_pixel(grown_pixels_x(b, a) + c, grown_pixels_y(b, a) + d, a);
                            counter_new_seed = counter_new_seed + 1;
                        end
                    end
                end
            end
        end
    end
    outimage = temp_image;
end