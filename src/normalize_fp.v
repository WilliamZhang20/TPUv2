module int18_to_bf16_lzd #(
    parameter FRAC_BITS = 8  // Number of fractional bits in Q10.8
)(
    input  wire signed [17:0] acc,
    output reg  [15:0] bf16
);
    localparam BF16_BIAS = 127;
    
    reg sign;
    reg [17:0] mag;
    reg [4:0] lz;
    reg signed [8:0] exp_unbiased;
    reg [7:0] exp;
    reg [6:0] mant;
    reg [17:0] normalized;

    // Leading zero detector
    function [4:0] lzd;
        input [17:0] x;
        integer i;
        begin
            lzd = 5'd18;  // Default: all zeros
            for (i=17; i>=0; i=i-1) begin
                if (x[i]==1'b1) begin
                    lzd = 5'd17 - i;
                    i = -1;  // Break
                end
            end
        end
    endfunction

    always @(*) begin
        sign = acc[17];
        mag  = sign ? -acc : acc;

        // Default values
        bf16 = 16'h0;

        if (mag != 0) begin
            lz = lzd(mag);
            
            // MSB position is (17 - lz), binary point is at FRAC_BITS
            // Unbiased exponent = MSB_position - FRAC_BITS
            exp_unbiased = (17 - lz) - FRAC_BITS;
            
            // Biased exponent for BF16
            if (exp_unbiased + BF16_BIAS < 0) begin
                // Underflow to zero
                bf16 = {sign, 15'd0};
            end else if (exp_unbiased + BF16_BIAS > 255) begin
                // Overflow to infinity
                bf16 = {sign, 8'hFF, 7'd0};
            end else begin
                exp = exp_unbiased + BF16_BIAS;
                
                // Normalize: shift to put MSB at bit 17, then extract bits [17:11] (drop the implicit 1)
                normalized = mag << (lz + 1);
                mant = normalized[17:11];
                
                bf16 = {sign, exp, mant};
            end
        end
    end
endmodule