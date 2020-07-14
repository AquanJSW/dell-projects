`timescale 1ns/1ns

module compare8_t;
reg[7:0] a,b;
reg clock;
wire re,reb,eq;
integer i;
initial
begin
    clock=0;
    a=8'b00000000;
    b=8'b00000000;
end

always #50 clock=~clock;
always@(posedge clock)
begin
    for(i=0;i<8;i=i+1)
        a[i]={$random}%2;
    for(i=0;i<8;i=i+1)
        b[i]={$random}%2;
end
initial
begin #1000 $stop; end
compare8 u0(.a(a), .b(b), .re(re), .reb(reb), .eq(eq));
endmodule
