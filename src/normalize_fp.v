// ----------------------------------------------------------------------------
// Microscaling Accumulator with BF16 output
// ----------------------------------------------------------------------------
module mx_accumulator #(
    parameter int PROD_WIDTH = 12
)(
    input  wire prod_sign,
    input  wire [8:0] prod_exp,
    input  wire [PROD_WIDTH-1:0] prod_mant,
    input  wire [15:0] acc_in,
    input  wire new_block,

    output wire [15:0] acc_out
);

    // Unpack incoming BF16 accumulator value
    wire acc_sign = acc_in[15];
    wire [7:0] acc_exp = acc_in[14:7];
    wire [6:0] acc_frac = acc_in[6:0];

    // Normalize mantissa
    wire prod_overflow = prod_mant[PROD_WIDTH-1];
    wire [8:0]  prod_exp_norm  = prod_overflow ? prod_exp + 9'd1 : prod_exp;
    wire [PROD_WIDTH-1:0] prod_mant_norm =
        prod_overflow ? prod_mant : (prod_mant << 1);

    // Convert product to BF16
    wire [15:0] prod_bf16;
    mx_to_bf16_converter #(.PROD_WIDTH(PROD_WIDTH)) converter (
        .sign(prod_sign),
        .exp(prod_exp_norm),
        .mant(prod_mant_norm),
        .bf16_out(prod_bf16)
    );

    // Simple BF16 add
    bf16_adder_simple adder (
        .a(acc_in),
        .b(prod_bf16),
        .sum(acc_out)
    );
endmodule
