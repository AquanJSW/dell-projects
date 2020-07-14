// 秒表

`timescale 1ms/1ms

module stopwatch(   output[5:0] min, sec,
                    output[6:0] subsec,         // 百分秒
                    input       reset, pause);
reg[5:0] min_reg, sec_reg;
reg[6:0] subsec_reg;
reg clk;

assign min = min_reg;
assign sec = sec_reg;
assign subsec = subsec_reg;

initial
begin
    min_reg = 0;
    sec_reg = 0;
    clk = 1;
    subsec_reg = 0; 
end


always #5 clk = !clk;


always @(posedge clk, reset, pause)
    if(reset)   // 异步置位
    begin
        min_reg = 0;
        sec_reg = 0;
        subsec_reg = 0;
    end

    else if(pause);  // 异步暂停

    else
        if(subsec_reg != 99)
            subsec_reg = subsec_reg + 1;
        
        else
            if(sec_reg != 59)   // 百分秒进位
            begin
                subsec_reg = 0;
                sec_reg = sec_reg + 1;
            end
            
            else
                if(min_reg != 59)   // 百分秒、秒进位
                begin
                    subsec_reg = 0;
                    sec_reg = 0;
                    min_reg = min_reg + 1;
                end

                else    // 百分秒、秒、分进位
                begin
                    subsec_reg = 0;
                    sec_reg = 0;
                    min_reg = 0;
                end
endmodule