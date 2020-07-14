module deside #(parameter PERSON=3) // 人数，默认为3
               (output               out, even,
                input[PERSON-1: 0]   in,
                input                reset);

reg out_reg, even_reg;
reg [PERSON-1: 0]i, ycount, ncount;

initial
begin
    i = 0;
    ycount = 0;
    ncount = 0;
    even_reg = 1;
end

// 组合电路不用非阻塞赋值
always@(in)
begin
    if(!reset)
    begin
        // 判断人数的奇偶
        if(PERSON % 2)
        begin
            i = 0;
            ycount = 0;
            ncount = 0;
            even_reg = 0;
            // 统计 通过/否决 表决数
            while(i < PERSON)
            begin
                if(in[i])
                    ycount = ycount + 1;
                else if (!in[i])
                    ncount = ncount + 1;
                i = i + 1;
            end

            // 给出判断结果
            if(ycount > ncount)
                out_reg = 1;
            else
                out_reg = 0;
        end

        else
            even_reg = 1;   // 人数为偶
    end
end

assign out = out_reg;
assign even = even_reg;
endmodule