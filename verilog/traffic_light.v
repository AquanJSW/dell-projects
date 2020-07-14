/* 
交通信号灯

    G   ___40___                    L
E   Y           5   5               O 
W   R                _____55____    O
    L            15_                P

    G                __30__         L
S   Y                      5   5    O
N   R   ______65_____               O
    L                       15_     P
        -----------------------> time
*/

`timescale 1ms/1ms
module traffic_light(output[1:0] G, Y, R, L,    // 高位东西，低位南北
                     input  rst);

reg[1:0] G_reg, Y_reg, R_reg, L_reg;
reg[6:0] i; // 计数器
reg clk;    // 时钟

assign G = G_reg;
assign Y = Y_reg;
assign R = R_reg;
assign L = L_reg;

// 初始化
initial
begin
    i = -1;
    G_reg = 0;
    Y_reg = 0;
    R_reg = 0;
    L_reg = 0;
    clk = 0;
end

// T = 1s
always#500 clk = ~clk;

// 时钟上升沿触发计数
always@ (posedge clk, rst)
begin
    if(!rst)    // rst低电平计数
    begin
        i = i + 1;

        // if...else if...实现交通信号灯的变化
        if(i >= 0 && i < 40)
        begin
            G_reg[1] = 1;
            R_reg[0] = 1;
        end

        else if(i >= 40 && i < 45)
        begin
            G_reg[1] = 0;
            Y_reg[1] = 1;
        end

        else if(i >= 45 && i < 60)
        begin
            Y_reg[1] = 0;
            L_reg[1] = 1;
        end

        else if(i >= 60 && i < 65)
        begin
            L_reg[1] = 0;
            Y_reg[1] = 1;
        end

        else if(i >= 65 && i < 95)
        begin
            Y_reg[1] = 0;
            R_reg[0] = 0;
            R_reg[1] = 1;
            G_reg[0] = 1;
        end

        else if(i >= 95 && i < 100)
        begin
            G_reg[0] = 0;
            Y_reg[0] = 1;
        end

        else if(i >= 100 && i < 115)
        begin
            Y_reg[0] = 0;
            L_reg[0] = 1;
        end

        else if(i >= 115 && i < 120)
        begin
            L_reg[0] = 0;
            Y_reg[0] = 1;
        end

        else if(i == 120)
        begin
            Y_reg = 0;
            L_reg = 0;
            G_reg = 2'b10;
            R_reg = 2'b01;
            i = 0;
        end
    end

    else    // rst高电平置位
    begin
        i = -1;
        G_reg = 0;
        R_reg = 0;
        Y_reg = 0;
        L_reg = 0;
    end
end
endmodule


