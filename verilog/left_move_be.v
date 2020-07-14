module left_move_be(reset, in, out, clock);
input in, reset, clock;
output[3:0] out;
reg[3:0] out;

always@(reset or posedge clock)
begin
    if(reset == 1)
        out <= 4'b0000;
    else
        out <= {out[2:0], in};
end
endmodule