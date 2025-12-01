module mxfp8_mul_int (
    input  wire       sign_a,
    input  wire       sign_b,
    input  wire [3:0] mant_a,
    input  wire [3:0] mant_b,
    input  wire [3:0] exp_a,
    input  wire [3:0] exp_b,

    output wire        prod_sign,
    output wire [7:0]  shift_amount,   // shift used for accumulation scaling
    output wire [7:0]  mant_prod       // integer product
);

    localparam int FP8_BIAS = 7;
    localparam int INT_ACC_OFFSET = 24;

    assign prod_sign = sign_a ^ sign_b;

    // Integer mantissa product: 4×4 = 8 bits
    assign mant_prod = mant_a * mant_b;

    // Compute shift amount into accumulator domain
    //   shift = exp_a + exp_b - F8_bias + INT_ACC_OFFSET
    assign shift_amount = exp_a + exp_b - FP8_BIAS + INT_ACC_OFFSET;

endmodule
