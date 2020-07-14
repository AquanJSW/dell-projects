/*
自助售票机
1 选择票价2-10元，整数
2 投币只能1元、5元或10元（这里不实现判断功能）
3 找零（实现1元、5元和10元混合找零，假设机器中各种面额钱币数量充足）
*/

module seller(output[3:0] ticket_out,   // 范围2-10，所出票的类型标志，也可以当做所出票的价格；当钱不足以支付时，输出0
              output[3:0] one_out, five_out, ten_out,   // 找零计数器
              input[3:0] one, five, ten, // 三种钱币计数器
              input[3:0] ticket);   // 所选票，范围2-10

reg[3:0] ticket_out_reg;
reg[3:0] one_out_reg, five_out_reg, ten_out_reg;
reg[7:0] money, money_left; // 分别是所投金币总额和每次找零后所剩余金币总额

assign ticket_out = ticket_out_reg;
assign one_out = one_out_reg;
assign five_out = five_out_reg;
assign ten_out = ten_out_reg;

always@(one, five, ten, ticket)
begin
    // 选票正确
    if(ticket >= 2 && ticket <= 10)
    begin
        money = one + five * 5 + ten * 10;  // 投的金币总额
        money_left = money - ticket;

        // 钱不够
        if(money_left < 0)
        begin
            ticket_out_reg = 0; // 不出票
            // 全额退回
            one_out_reg     = one;
            five_out_reg    = five;
            ten_out_reg     = ten;
        end

        // // 钱刚好够
        // else if(money_left == 0)
        // begin
        //     ticket_out_reg = ticket;
        //     one_out_reg     = 0;
        //     five_out_reg    = 0;
        //     ten_out_reg     = 0;
        // end

        // 钱够
        else
        begin
            ticket_out_reg = ticket;
            // 先找10块
            ten_out_reg = money_left / 10;   // 找多少10元
            money_left = money_left % 10; // 还剩多少
            // 再找5块
            five_out_reg = money_left / 5;
            money_left = money_left % 5;
            // 最后找1块
            one_out_reg = money_left;
        end
    end

    // 选票不正确
    else
    begin
        ticket_out_reg = 0;
        one_out_reg     = one;
        five_out_reg    = five;
        ten_out_reg     = ten;
    end
end

endmodule
