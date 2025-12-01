module int18_to_bf16_lzd (
    input  wire signed [17:0] acc,
    output reg  [15:0] bf16
);
    localparam BF16_BIAS = 127;
    reg sign;
    reg [17:0] mag;
    reg [4:0] lz;
    reg [7:0] exp;
    reg [6:0] mant;

    // Optimized tree-based LZD
    function [4:0] lzd;
        input [17:0] x;
        integer i;
        begin
            lzd = 5'd0;
            for (i=17; i>=0; i=i-1) begin
                if (x[i]==1 && lzd==0) lzd = 5'd17 - i;
            end
        end
    endfunction

    always @(*) begin
        sign = acc[17];
        mag  = sign ? -acc : acc;

        // default values to avoid latches
        lz   = 5'd0;
        exp  = 8'd0;
        mant = 7'd0;
        bf16 = 16'h0;

        if (mag != 0) begin
            lz   = lzd(mag);
            exp  = (18 - lz) + (BF16_BIAS - 1); 
            if (lz < 11)
                mant = (mag << (lz+1)) >> 11;
            else
                mant = 7'd0;
            bf16 = {sign, exp, mant};
        end
    end
endmodule
