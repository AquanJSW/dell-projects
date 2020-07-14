%开始生长
function grow_start(inimage, T)
    global grown_pixels_x;
    global grown_pixels_y;
    global region_num;
    global region_size;
    
	%用于存储待测的像素(包括了种子像素和邻域像素)
    candidate_pixels_x = zeros(1, region_size * 9);
    candidate_pixels_y = zeros(1, region_size * 9);
    
    %种子像素灰度均值
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
end