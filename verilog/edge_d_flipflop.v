module edge_d_flipflop(d,cp,r,q); 
    input d,cp,r;
    output q;
    reg q;
    always@(posedge cp)
    begin 
        if(r)
            q <= 0;
        else
            q <= d;
    end
endmodule