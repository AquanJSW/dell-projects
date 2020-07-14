`timescale 1ns/1ns

module left_move_as_t();
reg in, clock, reset;
wire[3:0] out;

initial
begin
    in = 0;
    clock = 1'b0;
    reset = 1'b1;
    #100 reset = 1'b0;
end

always 
    #50 clock = ~clock;

always@(posedge clock)
begin
    #20 in <= {$random}%2;
end

initial
begin
    #555 reset = 1'b1;
    #100 reset = 1'b0;
end

left_move_as u0(.in(in), .clock(clock), .reset(reset), .out(out));
endmodule