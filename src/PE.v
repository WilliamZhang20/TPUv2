module PE #(
    parameter WIDTH = 8
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    clear,

    input  wire [7:0]              a_in,   // FP8 E4M3
    input  wire [7:0]              b_in,   // FP8 E4M3

    output reg  [7:0]              a_out,
    output reg  [7:0]              b_out,
    output reg  [15:0]             c_out   // BF16 accumulator
);

    // ================================================================
    // 1. FP8 Decode → Integer mant, exp, sign
    // ================================================================
    wire sign_a = a_in[7];
    wire sign_b = b_in[7];
    wire sign_p = sign_a ^ sign_b;

    wire [3:0] exp_a  = a_in[6:3];
    wire [3:0] exp_b  = b_in[6:3];

    wire [3:0] mant_a = (exp_a == 0) ? 0 : {1'b1, a_in[2:0]}; // 8..15
    wire [3:0] mant_b = (exp_b == 0) ? 0 : {1'b1, b_in[2:0]};

    // 4×4 → 8-bit multiply
    wire [7:0] mant_prod_raw = mant_a * mant_b;

    // FP8 bias = 7
    wire [9:0] exp_sum_raw = exp_a + exp_b - 7;

    wire prod_zero = (mant_prod_raw == 0) | (exp_a == 0) | (exp_b == 0);

    // ================================================================
    // 2. Normalize INT mantissa product → BF16 format
    //       mant_prod_raw is 8–225
    //       => represent as 1.xxxxx BF16 mantissa (7 bits)
    // ================================================================
    reg  [7:0] mant_norm;
    reg  [7:0] exp_norm;

    always @(*) begin
        if (prod_zero) begin
            mant_norm = 0;
            exp_norm  = 0;
        end
        else if (mant_prod_raw[7]) begin
            // 1xx.xxxxx range → shift down 1 bit
            // mantissa: top 7 bits
            mant_norm = mant_prod_raw[7:1]; 
            exp_norm  = exp_sum_raw + 127 + 1;
        end 
        else begin
            // 0xx.xxxxx range → already normalized
            mant_norm = mant_prod_raw[6:0];
            exp_norm  = exp_sum_raw + 127;
        end
    end

    // ================================================================
    // 3. Reconstruct BF16 product
    // ================================================================
    wire [15:0] bf16_prod = {sign_p, exp_norm, mant_norm};

    // ================================================================
    // 4. BF16 adder (your original one)
    // ================================================================
    function automatic [15:0] bf16_add(
        input [15:0] a,
        input [15:0] b
    );
        reg signa, signb, signr;
        reg [7:0] expa, expb, expr, ediff;
        reg [9:0] ma, mb, ms;
        begin
            signa = a[15]; expa = a[14:7]; ma = (expa==0)? 0 : {1'b1,a[6:0],1'b0};
            signb = b[15]; expb = b[14:7]; mb = (expb==0)? 0 : {1'b1,b[6:0],1'b0};

            if (expa > expb) begin
                ediff = expa - expb; expr = expa;
                mb = mb >> ediff;
            end else begin
                ediff = expb - expa; expr = expb;
                ma = ma >> ediff;
            end

            if (signa == signb) begin
                ms = ma + mb;
                signr = signa;
            end else if (ma >= mb) begin
                ms = ma - mb;
                signr = signa;
            end else begin
                ms = mb - ma;
                signr = signb;
            end

            if (ms[9]) begin
                bf16_add = {signr, expr+1, ms[9:3]};
            end else if (ms[8]) begin
                bf16_add = {signr, expr, ms[8:2]};
            end else begin
                bf16_add = 16'h0000;
            end
        end
    endfunction

    // ================================================================
    // 5. Pipeline + BF16 accumulation (unchanged)
    // ================================================================
    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;

        if (rst)
            c_out <= 16'd0;
        else if (clear)
            c_out <= bf16_prod;
        else
            c_out <= bf16_add(c_out, bf16_prod);
    end

endmodule
