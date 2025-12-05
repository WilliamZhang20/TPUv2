module int18_to_bf16_lzd #(
    parameter FRAC_BITS = 8
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
    
    // ===================== HIERARCHICAL LEADING ZERO DETECTOR =====================
    wire [2:0] lz_hi, lz_mid, lz_lo;
    wire nz_hi, nz_mid, nz_lo;
    
    lzd6 lzd_hi  (.x(mag[17:12]), .lz(lz_hi),  .nz(nz_hi));
    lzd6 lzd_mid (.x(mag[11:6]),  .lz(lz_mid), .nz(nz_mid));
    lzd6 lzd_lo  (.x(mag[5:0]),   .lz(lz_lo),  .nz(nz_lo));
    
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
            // Combine hierarchical LZD results with proper offsets
            if (nz_hi)
                lz = {2'b00, lz_hi};           // bits [17:12]: offset 0
            else if (nz_mid)
                lz = 5'd6 + {2'b00, lz_mid};   // bits [11:6]: offset 6
            else if (nz_lo)
                lz = 5'd12 + {2'b00, lz_lo};   // bits [5:0]: offset 12
            else
                lz = 5'd18;
            
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

// ===================== 6-BIT LEADING ZERO DETECTOR =====================
module lzd6 (
    input  wire [5:0] x,
    output reg  [2:0] lz,
    output wire nz  // Non-zero flag
);
    assign nz = |x;
    
    always @(*) begin
        casez (x)
            6'b1?????: lz = 3'd0;
            6'b01????: lz = 3'd1;
            6'b001???: lz = 3'd2;
            6'b0001??: lz = 3'd3;
            6'b00001?: lz = 3'd4;
            6'b000001: lz = 3'd5;
            default:   lz = 3'd6;
        endcase
    end
endmodule