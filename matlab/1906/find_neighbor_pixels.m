%Ѱ��������Χ��������
%�����seed_num��ʾԭʼ���ӱ��
function [neighbor_pixels_x, neighbor_pixels_y, neighbor_pixels_num] ...
          = find_neighbor_pixels(image_size, seed_num)
    global grown_pixels_x;
    global grown_pixels_y;
    global region_size;
    
    %�洢�˴��������(�������������غ���������)
    candidate_pixels_x = zeros(1, region_size * 9);
    candidate_pixels_y = zeros(1, region_size * 9);
    
    %�������ػҶȾ�ֵ
    mean_intens_seed = 0;
    for a = 1 : region_size
        mean_intens_seed = mean_intens_seed + 
    end
    
    for a = 1 : region_size
        flag = 1;
        for b = -1 : 1
            for c = -1 : 1
                candidate_pixels_x(flag + a * 9) ...
                    = grown_pixels_x(a, seed_num) + b;
                candidate_pixels_y(flag + a * 9) ...
                    = grown_pixels_y(a, seed_num) + c;
                flag = flag + 1;
            end
        end
    end
    
    neighbor_pixels_num = 0;
    for a = 1 : region_size * 9
        if candidate_pixels_x(a) > 0 && candidate_pixels_y(a) > 0 ...
                && candidate_pixels_x(a) < image_size ...
                && candidate_pixels_y(a) < image_size
            if candidate_pixels_x(a) == 
        end
    end
end