module div(out, clk, n, reset);

input[3: 0]n;
input clk, reset;
output out;
reg odiv0, odiv1, ediv; 
reg[3: 0]ediv_flag;
reg[4: 0]odiv0_flag, odiv1_flag;
initial 
begin
    ediv_flag = 0;
    odiv0 = 0;
    odiv0_flag = 0;
    odiv1 = 0;
    odiv1_flag = 0;
end

// 上升沿触发的奇数分频器 & 偶数分频器
always@(posedge clk or posedge reset)
begin
    if (reset)
    begin
         ediv <= 0;
         odiv1 <= 0;
    end

    else
        if(n >= 4'b0010 & n <= 4'b1111)
        begin
            // 奇数分频
            if (n[0])
            begin
                if(odiv1_flag == 0 | (odiv1_flag*2-1) == n)
                    odiv1 <= ~odiv1;
                odiv1_flag <= odiv1_flag + 1;
                if (odiv1_flag == n)
                begin
                    odiv1 <= ~odiv1;
                    odiv1_flag <= 1;
                end
            end

            // 偶数分频
            else
            begin
                if(ediv_flag == 0 | ediv_flag == n/2)
                    ediv <= ~ediv;

                ediv_flag <= ediv_flag + 1;

                if(ediv_flag == n)
                begin
                    ediv <= ~ediv;
                    ediv_flag <= 1;
                end
            end
        end
end

// 下降沿触发的奇数分频器
always@(negedge clk or posedge reset)
begin
    if (reset)
        odiv0 <= 0;
    else
    begin
        if (n >= 4'b0010 & n <= 4'b1111)
        begin
            if (n[0])
            begin
                if (odiv0_flag == 0 | (odiv0_flag*2-1) == n)
                    odiv0 <= ~odiv0;
                odiv0_flag <= odiv0_flag + 1;
                if (odiv0_flag == n)
                begin
                    odiv0 <= ~odiv0;
                    odiv0_flag <= 1;
                end
            end
        end
    end
end

or(out, odiv0 & odiv1, ediv);
endmodule