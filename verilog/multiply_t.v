// 8位二进制乘法器测试

`timescale 1ns/1ns

module multiply_t();

reg[7: 0] in0, in1;
reg reset;
reg[3: 0] i;
wire[15: 0] out;

initial
begin
    in0 = 0;
    in1 = 0;
    reset = 1;
    #50 reset = 0;
end

// 随机输入
always#100
begin
    for(i=0; i<8; i=i+1)
    begin
        in0[i] = {$random} % 2;
        in1[i] = {$random} % 2;
    end
end

multiply_for U0(.out(out), .in0(in0), .in1(in1), .reset(reset));
multiply_repeat U1(.out(out), .in0(in0), .in1(in1), .reset(reset));

endmodule