`timescale 1ns/1ns

module div_t();

reg clk, reset;
reg[3: 0]n;
wire out;

initial
begin
    clk = 0;
    reset = 1;
    # 50 reset = 0;
    n = 4'd3;
end

always
    #50 clk = ~clk;

div u0(.clk(clk), .n(n), .reset(reset), .out(out));

endmodule