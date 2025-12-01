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

    // Stage 1 pipeline registers
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

    // -----------------------
    // Stage 2: Multiply + Align (combined)
    // -----------------------
    wire prod_sign_s2;
    wire [17:0] aligned_prod_s2; // reduced width accumulator target
    assign prod_sign_s2 = sign_a_s1 ^ sign_b_s1;

    // Combined multiply + align
    wire [7:0] mant_prod_s2 = mant_a_s1 * mant_b_s1;
    localparam int FP8_BIAS = 7;
    localparam int INT_ACC_OFFSET = 18; // reduce width from 20 -> 18
    wire [7:0] shift_amount_s2 = exp_a_s1 + exp_b_s1 - FP8_BIAS + INT_ACC_OFFSET;

    assign aligned_prod_s2 = mant_prod_s2 << shift_amount_s2;

    // -----------------------
    // Stage 3: Accumulate INT18
    // -----------------------
    reg signed [17:0] acc_s3;

    always @(posedge clk) begin
        if (rst || clear) acc_s3 <= 0;
        else acc_s3 <= prod_sign_s2 ? acc_s3 - aligned_prod_s2 : acc_s3 + aligned_prod_s2;
    end

    // -----------------------
    // Stage 4: INT → BF16 (optimized LZD)
    // -----------------------
    wire [15:0] c_val;
    int18_to_bf16_lzd convert_bf16 (
        .acc(acc_s3),
        .bf16(c_val)
    );

    always @(posedge clk) begin
        if (rst) c_out <= 16'h0;
        else     c_out <= c_val;
    end

endmodule
