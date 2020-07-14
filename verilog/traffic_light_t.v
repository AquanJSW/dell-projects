// 交通信号灯测试

`timescale 1ms/1ms
module traffic_light_t();

reg rst;
wire[1:0] R, L, Y, G;

initial
begin
    rst = 1; #600
    rst = 0;
end

traffic_light U0(.R(R), .L(L), .Y(Y), .G(G), .rst(rst));

endmodule