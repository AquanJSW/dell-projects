module compare8(a,b,re,reb,eq);
input[7:0] a,b;
output re,reb,eq;
reg re,reb,eq;

always@(a or b)
begin
    if(a==b)
    begin
        eq=1;
    end
    else
    begin
        eq=0;
    end
    if(a>b)
    begin
        re=1;
    end
    else
    begin
        re=0;
    end
    if(a<b)
    begin
        reb=1;
    end
    else
    begin
        reb=0;
    end
end
endmodule
