module compare8_assign(a,b,re,reb,eq);
input[7:0] a,b;
output re,reb,eq;

assign re=(a>b);
assign reb=(a<b);
assign eq=(a==b);
endmodule
