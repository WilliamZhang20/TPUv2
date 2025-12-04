module int18_to_bf16_lzd #(
    parameter FRAC_BITS = 8  // Number of fractional bits in Q10.8
)(
    input  wire signed [17:0] acc,
    output reg  [15:0] bf16
);
    localparam BF16_BIAS = 8'd127;
    
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
        reg found;
        begin
            lzd   = 5'd18;
            found = 1'b0;

            for (i = 17; i >= 0; i = i - 1) begin
                if (!found && x[i]) begin
                    lzd   = 5'd17 - i[4:0];  // avoid width warnings
                    found = 1'b1;            // emulate break
                end
            end
        end
    endfunction

    always @(*) begin
        // Default assignments (prevent latches)
        sign        = acc[17];
        mag         = sign ? -acc : acc;
        bf16        = 16'h0;

        lz          = 5'd0;
        exp_unbiased= 9'sd0;
        exp         = 8'd0;
        mant        = 7'd0;
        normalized  = 18'd0;

        if (mag != 0) begin
            lz = lzd(mag);

            // MSB position is (17 - lz), binary point at FRAC_BITS
            exp_unbiased = $signed({1'b0, (9'd17 - {4'd0,lz})}) - FRAC_BITS;

            if ($signed(exp_unbiased) + $signed(BF16_BIAS) < 0) begin
                bf16 = {sign, 15'd0}; // underflow
            end else if ($signed(exp_unbiased) + $signed(BF16_BIAS) > 9'd255) begin
                bf16 = {sign, 8'hFF, 7'd0}; // overflow
            end else begin
                exp = exp_unbiased[7:0] + BF16_BIAS;

                // normalize so MSB ends up at bit 17
                normalized = mag << (lz + 5'd1);
                mant = normalized[17:11];

                bf16 = {sign, exp, mant};
            end
        end
    end
endmodule
