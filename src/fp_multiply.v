module bf16_multiplier (
    input  logic        signA, signB,
    input  logic [7:0]  expA,  expB,
    input  logic [7:0]  mantA, mantB,     // includes hidden bit
    output logic        prod_zero,
    output logic        prod_inf,
    output logic        prod_sign,
    output logic [8:0]  prod_exp,
    output logic [15:0] prod_mant
);

    always @(*) begin
        prod_sign = signA ^ signB;
        prod_zero = (mantA == 0) || (mantB == 0);
        prod_inf  = 1'b0;

        if (prod_zero) begin
            prod_exp  = 0;
            prod_mant = 0;
        end else begin
            prod_mant = mantA * mantB;                    // 8x8 → 16-bit exact
            prod_exp  = expA + expB - 127;                // BF16 bias = 127
        end
    end
endmodule