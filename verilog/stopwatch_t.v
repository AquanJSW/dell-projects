// 秒表测试

`timescale 1ms/1ms

module stopwatch_t();

reg reset, pause;
wire[5:0] min, sec;
wire[6:0] subsec;

initial
begin
    reset = 0;
    pause = 0;
    #1234 reset = 1;
    #22 reset = 0;
    #1231 pause = 1;
    #22 pause = 0;
end

stopwatch U0(.min(min), .sec(sec), .subsec(subsec),
             .reset(reset), .pause(pause));

endmodule