// 自助售票机测试

`timescale 1s/1s
module seller_t();

reg[3:0] one, five, ten;
reg[3:0] ticket;
wire[3:0] one_out, five_out, ten_out;
wire[3:0] ticket_out;

initial
begin
    ticket = 7;
    one = 2;
    five = 1;
    ten = 0;
end

always#10
begin
    ticket = {$random} % 9 + 2;
    one = {$random} % 16;
    five = {$random} % 16;
    ten = {$random} % 16;
end

seller U0(.ticket_out(ticket_out), .one_out(one_out), .five_out(five_out), .ten_out(ten_out),
          .ticket(ticket), .one(one), .five(five), .ten(ten));

endmodule