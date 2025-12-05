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
        reg [17:0] scan;
        begin
            // Priority encode from MSB to LSB
            scan = x;
            
            casez (scan)
                18'b1?????????????????: lzd = 5'd0;
                18'b01????????????????: lzd = 5'd1;
                18'b001???????????????: lzd = 5'd2;
                18'b0001??????????????: lzd = 5'd3;
                18'b00001?????????????: lzd = 5'd4;
                18'b000001????????????: lzd = 5'd5;
                18'b0000001???????????: lzd = 5'd6;
                18'b00000001??????????: lzd = 5'd7;
                18'b000000001?????????: lzd = 5'd8;
                18'b0000000001????????: lzd = 5'd9;
                18'b00000000001???????: lzd = 5'd10;
                18'b000000000001??????: lzd = 5'd11;
                18'b0000000000001?????: lzd = 5'd12;
                18'b00000000000001????: lzd = 5'd13;
                18'b000000000000001???: lzd = 5'd14;
                18'b0000000000000001??: lzd = 5'd15;
                18'b00000000000000001?: lzd = 5'd16;
                18'b000000000000000001: lzd = 5'd17;
                default:                lzd = 5'd18;  // All zeros
            endcase
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
            exp_unbiased = 9'(17) - 9'(lz) - 9'(FRAC_BITS);
            
            if (exp_unbiased + BF16_BIAS < 0) begin
                bf16 = {sign, 15'd0}; // underflow
            end else if (exp_unbiased + BF16_BIAS > 255) begin
                bf16 = {sign, 8'hFF, 7'd0}; // overflow
            end else begin
                exp = exp_unbiased + BF16_BIAS;

                // normalize so MSB ends up at bit 17
                normalized = mag << lz;
                mant = normalized[16:10];
                bf16 = {sign, exp, mant};
            end
        end
    end
endmodule