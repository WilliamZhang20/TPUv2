// ----------------------------------------------------------------------------
// Microscaling Multiplier: (sign, mant) × (sign, mant) with block exponents
// ----------------------------------------------------------------------------
module mx_multiplier #(
    parameter int MANT_WIDTH = 5
)(
    input  wire sign_a,
    input  wire sign_b,
    input  wire [MANT_WIDTH-1:0] mant_a,
    input  wire [MANT_WIDTH-1:0] mant_b,
    input  wire [7:0] exp_a,
    input  wire [7:0] exp_b,

    output wire prod_sign,
    output wire [8:0] prod_exp,
    output wire [2*MANT_WIDTH+1:0] prod_mant
);
    localparam int BF16_BIAS = 127;

    // Sign: XOR
    assign prod_sign = sign_a ^ sign_b;

    // Mantissa multiply: (1.mant_a × 1.mant_b)
    wire [MANT_WIDTH:0] mant_a_hidden = {1'b1, mant_a};
    wire [MANT_WIDTH:0] mant_b_hidden = {1'b1, mant_b};
    assign prod_mant = mant_a_hidden * mant_b_hidden;

    // Exponent math: exp_a + exp_b - bias
    assign prod_exp = exp_a + exp_b - BF16_BIAS;
endmodule


// ----------------------------------------------------------------------------
// Convert normalized MX product to BF16
// ----------------------------------------------------------------------------
module mx_to_bf16_converter #(
    parameter int PROD_WIDTH = 12
)(
    input  wire sign,
    input  wire [8:0] exp,
    input  wire [PROD_WIDTH-1:0] mant,

    output reg  [15:0] bf16_out
);

    // Local temps must be REGs for use inside always blocks
    reg [7:0] mant_extended;
    reg       round_bit;
    reg [7:0] mant_rounded;

    always @(*) begin
        
        // Zero or underflow
        if (mant == 0 || exp[8]) begin
            bf16_out = {sign, 15'b0};
        end
        
        // Overflow → Inf
        else if (exp >= 9'd255) begin
            bf16_out = {sign, 8'hFF, 7'b0};
        end
        
        else begin

            if (PROD_WIDTH >= 9) begin
                // Equivalent to your earlier "wire = slice" statements
                mant_extended = mant[PROD_WIDTH-2 -: 8];

                if (PROD_WIDTH >= 10)
                    round_bit = mant[PROD_WIDTH-9];
                else
                    round_bit = 1'b0;

                mant_rounded = mant_extended + round_bit;

                // Detect carry out of rounding
                if (mant_rounded[7]) begin
                    bf16_out = {sign, exp[7:0] + 8'd1, 7'b0};
                end 
                else begin
                    bf16_out = {sign, exp[7:0], mant_rounded[6:0]};
                end
            end
            
            else begin
                bf16_out = {
                    sign,
                    exp[7:0],
                    mant[PROD_WIDTH-2:0], 
                    {(7 - (PROD_WIDTH-1)){1'b0}}
                };
            end
        end
    end
endmodule
