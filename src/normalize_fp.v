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

    lzd18 lzd_inst (.x(mag), .lz(lz));

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

module lzd18 (
    input  wire [17:0] x,
    output reg  [4:0] lz 
);

    integer i;
    always @(*) begin
        lz = 5'd18; // Default to 18 if all zeros
        for (i = 0; i < 18; i = i + 1) begin
            if (x[17 - i] == 1'b1) begin
                lz = i[4:0];
                i=17;
            end
        end
    end
endmodule