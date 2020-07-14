`define PERSON 6   // 全局参数
`timescale 1ns/1ns

module deside_t();

reg [`PERSON-1:0]in;
reg [`PERSON-1:0]i;
reg reset;
wire out, even;

// 初始化
initial
begin
    in = 0;
    reset = 1;
    #50 reset = 0;
end

// 每隔100ns就生产一次随机决策
always
begin
    #100
    i = 0;
    while(i < `PERSON)
    begin
        in[i] = {$random} % 2;
        i = i + 1;
    end
end

deside #(.PERSON(`PERSON))
        U0(.out(out), .even(even), .in(in), .reset(reset));

endmodule