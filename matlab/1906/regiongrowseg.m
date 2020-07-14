%�����������ָ�ͼ��
%T=0~1,��Ϊֹͣ��������ֵ��Խ���ʾ����Խ�٣�0��ʾ���������ӻҶ�ֵ��ͬ��1��ʾ
%�����ƣ���ʱ�����壩��Ĭ�����0.5
%����������ʹ�ö�ֵ��ʾ
%ע�⣬��inimage(a,b)��[x,y]=ginput������x��Ӧb,y��Ӧa
%inimage(grown_pixels_x(b, a) + c, grown_pixels_y(b, a) + d) ��ʽ�����κ�����
%����ǿ�н����ת����0~255֮�ڣ�����Ҫ��double
function outimage = regiongrowseg(inimage, T)
    %����Ĭ����ֵ
    if nargin == 1
        T = 0.5;
    end
    T = T * 255;
    
    %���ȡ���ɸ�����Ϊ����(seed)���س�����
    global seedy;
    global seedx;
    imshow(inimage);
    [seedy, seedx] = ginput;
    seedx = round(seedx);
    seedy = round(seedy);
    
    global region_num;
    [region_num, ~] = size(seedx);%ԭʼ������Ŀ(������Ŀ)
    
    [lenx_inimage, leny_inimage] = size(inimage);
    pixels_inimage = lenx_inimage * leny_inimage;%ͼ��������
    
    %�洢�����������������������ջ��grown_pixels_x, grown_pixels_y;
    global grown_pixels_x;
    global grown_pixels_y;
    grown_pixels_x = zeros(pixels_inimage, region_num);
    grown_pixels_y = zeros(pixels_inimage, region_num);
    
    %�洢��������������������
    global region_size;
    region_size = zeros(1, region_num);
    
    %��ֵ����ͼ��
    global temp_image;
    temp_image = zeros(lenx_inimage, leny_inimage);
    
    %�����������
    for a = 1 : region_num
        store_grown_pixel(seedx(a), seedy(a), a);%Ԫ����������ջ
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