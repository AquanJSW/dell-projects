%�洢�������������������ջ��grown_pixels_x, grown_pixels_y;
%�����seed_num��ʾԭʼ���ӱ��
function store_grown_pixel(x, y, seed_num)
    global grown_pixels_x;
    global grown_pixels_y;
    global region_size;
    global temp_image;
    
    temp_image(x, y) = 255;%������ͼ���ϵĸ����ӵ���255
    
    grown_pixels_x(region_size(seed_num) + 1, seed_num) = x;
    grown_pixels_y(region_size(seed_num) + 1, seed_num) = y;
    region_size(seed_num) = region_size(seed_num) + 1;
end