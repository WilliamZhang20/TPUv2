module PE (
    input  wire        clk,
    input  wire        rst,
    input  wire        clear,
    input  wire [7:0]  a_in,
    input  wire [7:0]  b_in,
    output reg  [7:0]  a_out,
    output reg  [7:0]  b_out,
    output reg  [15:0] c_out
);

    // Pass-through systolic hooks
    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;
    end

    // ============================================================
    // Stage 1: Decode
    // ============================================================
    wire sign_a, sign_b;
    wire [3:0] exp_a, exp_b;
    wire [3:0] mant_a, mant_b;

    mxfp8_e4m3_decoder dec_a(a_in, sign_a, exp_a, mant_a);
    mxfp8_e4m3_decoder dec_b(b_in, sign_b, exp_b, mant_b);

    // Stage 1 pipeline regs
    reg sign_a_s1, sign_b_s1;
    reg [3:0] exp_a_s1, exp_b_s1;
    reg [3:0] mant_a_s1, mant_b_s1;

    always @(posedge clk) begin
        if (rst) begin
            sign_a_s1 <= 0; sign_b_s1 <= 0;
            exp_a_s1  <= 0; exp_b_s1  <= 0;
            mant_a_s1 <= 0; mant_b_s1 <= 0;
        end else begin
            sign_a_s1 <= sign_a;
            sign_b_s1 <= sign_b;
            exp_a_s1  <= exp_a;
            exp_b_s1  <= exp_b;
            mant_a_s1 <= mant_a;
            mant_b_s1 <= mant_b;
        end
    end

    // ============================================================
    // Stage 2: Multiply + Exponent Combine
    // ============================================================
    wire prod_sign_s2;
    wire [7:0] shift_s2;
    wire [7:0] mant_prod_s2;

    mxfp8_mul_int mul (
        .sign_a(sign_a_s1), .sign_b(sign_b_s1),
        .mant_a(mant_a_s1), .mant_b(mant_b_s1),
        .exp_a(exp_a_s1),   .exp_b(exp_b_s1),
        .prod_sign(prod_sign_s2),
        .shift_amount(shift_s2),
        .mant_prod(mant_prod_s2)
    );

    // Aligner
    wire [19:0] aligned_prod_s2;
    mx_int_aligner aligner (
        .mant_prod(mant_prod_s2),
        .shift_amount(shift_s2),
        .aligned_val(aligned_prod_s2)
    );

    // Stage 2 pipeline registers (acc input)
    reg prod_sign_s3;
    reg [19:0] aligned_s3;

    always @(posedge clk) begin
        if (rst) begin
            prod_sign_s3 <= 0;
            aligned_s3   <= 0;
        end else begin
            prod_sign_s3 <= prod_sign_s2;
            aligned_s3   <= aligned_prod_s2;
        end
    end

    // ============================================================
    // Stage 3: Accumulate INT20
    // ============================================================
    wire signed [19:0] acc_val;

    mx_int_acc accumulator (
        .clk(clk), .rst(rst), .clear(clear),
        .aligned_val(aligned_s3),
        .prod_sign(prod_sign_s3),
        .acc(acc_val)
    );

    // ============================================================
    // Stage 4: Convert INT → BF16
    // ============================================================
    wire [15:0] c_val;

    int20_to_bf16 convert_bf16 (
        .acc(acc_val),
        .bf16(c_val)
    );

    // register output
    always @(posedge clk) begin
        if (rst) c_out <= 0;
        else     c_out <= c_val;
    end

endmodule
