// 8位二进制乘法器

module multiply_for(output[15: 0] out,
                    input[7: 0] in0, in1,
                    input reset);

reg[14: 0] buffer;  // 输入数据移位缓冲
reg[15: 0] out_reg;
reg[7: 0] in0_reg, in1_reg;
reg[3: 0] i;

always@(in0, in1, reset)
begin
    if(reset)
        out_reg = 0;
    else
    begin
        in0_reg = in0;
        in1_reg = in1;
        out_reg = 0;

        // for乘法器
        for(i=0; i<8; i=i+1)
        begin

            if(in1_reg[i])
            begin
                buffer = in0_reg;
                buffer = buffer << i;
                out_reg = out_reg + buffer;
            end
        end
    end
end

assign out = out_reg;

endmodule