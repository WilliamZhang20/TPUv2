module PE (
    input  wire        clk,
    input  wire        rst,
    input  wire        clear,
    input  wire [7:0]  a_in,
    input  wire [7:0]  b_in,
    output reg  [15:0] c_out
);

    // ----------------------------
    // Stage 1 decode outputs (wires)
    // ----------------------------
    wire       a_sign_dec, b_sign_dec;
    wire [7:0] a_exp_dec,  b_exp_dec;
    wire [7:0] a_mant_dec, b_mant_dec;

    fp8_to_bf16_decoder #(.FTZ_FP8(1)) dec_a (
        .fp8(a_in),
        .sign(a_sign_dec),
        .exp_out(a_exp_dec),
        .mant_hidden_out(a_mant_dec)
    );

    fp8_to_bf16_decoder #(.FTZ_FP8(1)) dec_b (
        .fp8(b_in),
        .sign(b_sign_dec),
        .exp_out(b_exp_dec),
        .mant_hidden_out(b_mant_dec)
    );

    // ----------------------------
    // Stage 1 → Stage 2 pipeline registers
    // ----------------------------
    reg        a_sign_s1, b_sign_s1;
    reg [7:0]  a_exp_s1,  b_exp_s1;
    reg [7:0]  a_mant_s1, b_mant_s1;

    // ----------------------------
    // Stage 2 multiply outputs
    // ----------------------------
    wire        prod_sign_mul;
    wire [8:0]  prod_exp_mul;
    wire [15:0] prod_mant_mul;

    bf16_multiplier mul (
        .signA(a_sign_s1),
        .signB(b_sign_s1),
        .expA(a_exp_s1),
        .expB(b_exp_s1),
        .mantA(a_mant_s1),
        .mantB(b_mant_s1),
        .prod_sign(prod_sign_mul),
        .prod_exp(prod_exp_mul),
        .prod_mant(prod_mant_mul)
    );

    // ----------------------------
    // Stage 2 → Stage 3 pipeline registers
    // ----------------------------
    reg        prod_sign_s2;
    reg [8:0]  prod_exp_s2;
    reg [15:0] prod_mant_s2;

    // ----------------------------
    // Stage 3 normalization
    // ----------------------------
    wire [15:0] prod_bf16;

    bf16_normalizer norm (
        .sign(prod_sign_s2),
        .exp_in(prod_exp_s2),
        .mant_raw(prod_mant_s2),
        .bf16_result(prod_bf16)
    );

    // ----------------------------
    // Adder input and output
    // ----------------------------
    wire [15:0] acc_next;

    bf16_adder adder (
        .a(c_out),
        .b(prod_bf16),
        .sum(acc_next)
    );

    // ================================================================
    // PIPELINE REGISTERS
    // ================================================================
    always @(posedge clk) begin
        if (rst) begin
            // Stage 1 registers
            a_sign_s1 <= 0;  a_exp_s1 <= 0;  a_mant_s1 <= 0;
            b_sign_s1 <= 0;  b_exp_s1 <= 0;  b_mant_s1 <= 0;

            // Stage 2 registers
            prod_sign_s2 <= 0;
            prod_exp_s2  <= 0;
            prod_mant_s2 <= 0;

            // Accumulator
            c_out <= 16'd0;
        end
        else begin
            // Stage 1 → Stage 2
            a_sign_s1 <= a_sign_dec;
            a_exp_s1  <= a_exp_dec;
            a_mant_s1 <= a_mant_dec;

            b_sign_s1 <= b_sign_dec;
            b_exp_s1  <= b_exp_dec;
            b_mant_s1 <= b_mant_dec;

            // Stage 2 → Stage 3
            prod_sign_s2 <= prod_sign_mul;
            prod_exp_s2  <= prod_exp_mul;
            prod_mant_s2 <= prod_mant_mul;

            // Accumulator update
            if (clear)
                c_out <= 16'd0;
            else
                c_out <= acc_next;
        end
    end

endmodule
