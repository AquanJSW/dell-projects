`timescale 1ns/1ns

module left_move_be_t();
reg in, clock, reset;
wire[3:0] out;

initial
begin
    in = 0;
    clock = 1'b0;
    reset = 1'b1;
    #50 reset = 1'b0;
end

always 
    #50 clock = ~clock;

always@(negedge clock)
begin
    in <= {$random}%2;
end

initial
begin
    #555 reset = 1'b1;
    #55 reset = 1'b0;
end
left_move_be u0(.in(in), .clock(clock), .reset(reset), .out(out));
endmodule