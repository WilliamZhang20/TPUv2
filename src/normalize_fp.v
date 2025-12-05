module int18_to_bf16_lzd #(
    parameter FRAC_BITS = 8
)(
    input  wire signed [17:0] acc,
    output reg  [15:0] bf16
);
    localparam BF16_BIAS = 127;
    
    wire sign;
    wire [17:0] mag;
    wire [4:0] lz;
    wire signed [8:0] exp_unbiased;
    reg [7:0] exp;
    wire [6:0] mant;
    wire [17:0] normalized;

    assign sign = acc[17];
    assign mag = sign ? -acc : acc;

    // ===================== HIERARCHICAL LEADING ZERO DETECTOR =====================
    wire [2:0] lz_hi, lz_mid, lz_lo;
    wire nz_hi, nz_mid, nz_lo;
    
    lzd6 lzd_hi  (.x(mag[17:12]), .lz(lz_hi),  .nz(nz_hi));
    lzd6 lzd_mid (.x(mag[11:6]),  .lz(lz_mid), .nz(nz_mid));
    lzd6 lzd_lo  (.x(mag[5:0]),   .lz(lz_lo),  .nz(nz_lo));

    assign lz = nz_hi  ? {2'b00, lz_hi} :
                nz_mid ? 5'd6 + {2'b00, lz_mid} :
                nz_lo  ? 5'd12 + {2'b00, lz_lo} :
                         5'd18;
    
    assign exp_unbiased = 9'(17) - 9'(lz) - 9'(FRAC_BITS);

    assign normalized = mag << lz;

    assign mant = normalized[16:10];
    
    always @(*) begin
        exp = 8'b0;
        if (mag == 18'd0) begin
            bf16 = {sign, 15'd0}; // Zero or Negative Zero
        end else begin
            if (exp_unbiased + BF16_BIAS < 0) begin
                bf16 = {sign, 15'd0}; // Underflow (flush to zero)
            end else if (exp_unbiased + BF16_BIAS > 255) begin
                bf16 = {sign, 8'hFF, 7'd0}; // Overflow (infinity)
            end else begin
                // Convert to biased exponent and assemble BF16
                exp = exp_unbiased + BF16_BIAS;
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