module left_move_as(in, out, clock, reset);
input in, reset, clock;
output[3:0] out;

edge_d_flipflop d0(.d(in),   .cp(clock), .r(reset), .q(out[0]));
edge_d_flipflop d1(.d(d0.q), .cp(clock), .r(reset), .q(out[1]));
edge_d_flipflop d2(.d(d1.q), .cp(clock), .r(reset), .q(out[2]));
edge_d_flipflop d3(.d(d2.q), .cp(clock), .r(reset), .q(out[3]));

endmodule