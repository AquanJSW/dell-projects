%8����������������ɢͼ��
%T=0~1,��Ϊֹͣ��������ֵ��Խ���ʾ����Խ�٣�0��ʾ���������ӻҶ�ֵ��ͬ��1��ʾ
%�����ƣ���ʱ�����壩
function outimage = regiongrow8(inimage, T)
    %�������꣬��Ϊn*1����
    global seedx;
    global seedy;
    
    global region_num;
    [region_num, ~] = size(seedx);%ԭʼ������Ŀ(������Ŀ)
    
    [lenx_image, leny_image] = size(image);
    pixels_image = lenx_image * leny_image;%ͼ��������
    
    %�洢�������ص�����
    global grown_pixels_x;
    global grown_pixels_y;
    grown_pixels_x = zeros(pixels_image, region_num);
    grown_pixels_y = zeros(pixels_image, region_num);
    
    %�洢��������������������
    global region_size;
    region_size = zeros(1, region_num);
    
    %��ֵ����ͼ��
    global temp_image;
    temp_image = inimage;
    temp_image = 0;
    
    %����Ԫ��������
    for a = 1 : region_num
        store_grown_pixel([seedx(a), seedy(a)], a);
    end
    

    
    %����
    grow_start(inimage, T);
    
    outimage = temp_image;
end