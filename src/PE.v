module PE #(
    parameter int WIDTH = 8
)(
    input  logic             clk,
    input  logic             rst,
    input  logic             clear,        // load new value into accumulator
    input  logic [7:0]       a_in,          // FP8 E4M3
    input  logic [7:0]       b_in,          // FP8 E4M3
    output logic [7:0]       a_out,
    output logic [7:0]       b_out,
    output logic [15:0]      c_out          // BF16 accumulator
);

    // ============================================================
    // 1. FP8 → sign, mant (4-bit with implicit 1), exp
    // ============================================================
    logic sign_p;
    logic [3:0] exp_a, exp_b;
    logic [3:0] mant_a, mant_b;
    logic [7:0] mant_prod;
    logic [8:0] exp_sum;       // 0..30 → 9 bits enough
    logic       is_zero;

    assign sign_p   = a_in[7] ^ b_in[7];
    assign exp_a    = a_in[6:3];
    assign exp_b    = b_in[6:3];
    assign mant_a   = exp_a != 0 ? {1'b1, a_in[2:0]} : 4'd0;
    assign mant_b   = exp_b != 0 ? {1'b1, b_in[2:0]} : 4'd0;

    assign mant_prod = mant_a * mant_b;                    // 0 or 64..225
    assign exp_sum   = exp_a + exp_b;                      // bias handled later
    assign is_zero   = (mant_prod == 0) || (exp_a == 0) || (exp_b == 0);

    // ============================================================
    // 2. Pipeline stage 1 → register everything
    // ============================================================
    logic        sign_p_r;
    logic [7:0]  mant_prod_r;
    logic [8:0]  exp_sum_r;
    logic        zero_r;

    always_ff @(posedge clk) begin
        a_out       <= a_in;
        b_out       <= b_in;

        sign_p_r    <= sign_p;
        mant_prod_r <= mant_prod;
        exp_sum_r   <= exp_sum;
        zero_r      <= is_zero;
    end

    // ============================================================
    // 3. Convert FP8×FP8 → exact BF16 product (with rounding)
    // ============================================================
    logic [7:0]  ] bf_exp;
    logic [6:0]  bf_mant;
    logic [15:0] bf_prod;

    always_comb begin
        if (zero_r) begin
            bf_exp  = 8'd0;
            bf_mant = 7'd0;
        end else begin
            // mant_prod_r ∈ [64,225]
            logic [7:0] mant_tmp;
            logic       round_up;

            if (mant_prod_r[7]) begin
                // 128..225 → shift right once
                mant_tmp = mant_prod_r;               // 1xxx_xxxx
                bf_exp   = exp_sum_r + 8'd120;         // -7 bias + 127 BF16 bias + 1 shift
            end else begin
                // 64..127 → no shift
                mant_tmp = mant_prod_r << 1;           // 1xxx_xxx.0
                bf_exp   = exp_sum_r + 8'd119;         // -7 + 127 + 0 shift
            end

            // Round-to-nearest-even (only 1 sticky bit)
            round_up = mant_tmp[0];
            {bf_mant, bf_exp} = bf_exp + round_up; } = {mant_tmp[7:1] + round_up, bf_exp};
        end
    end

    assign bf_prod = {sign_p_r, bf_exp, bf_mant};

    // ============================================================
    // 4. Registered BF16 adder → only ~40–50 LUTs
    // ============================================================
    logic [15:0] acc;
    logic [15:0] add_b;

    always_ff @(posedge clk) add_b <= bf_prod;

    // Simple registered BF16 adder (you can copy your original one, just make it combinational)
    logic [15:0] sum;
    always_comb sum = bf16_add(acc, add_b);

    always_ff @(posedge clk) begin
        if (rst)
            acc <= 16'd0;
        else if (clear)
            acc <= bf_prod;         // first MAC of new row/dot-product
        else
            acc <= sum;
    end

    assign c_out = acc;

    // ============================================================
    // BF16 adder
    // ============================================================
    function automatic logic [15:0] bf16_add(input [15:0] a, b);
        logic sa = a[15], sb = b[15];
        logic [7:0] ea = a[14:7], eb = b[14:7];
        logic [7:0] ma = ea ? {1'b1, a[6:0]} : 8'd0;
        logic [7:0] mb = eb ? {1'b1, b[6:0]} : 8'd0;
        logic [7:0] e_max, e_diff;
        logic [8:0] m_sum;
        logic       swap;

        // Align exponents
        if (ea > eb) begin
            e_max = ea; e_diff = ea - eb; mb = mb >> e_diff;
        end else begin
            e_max = eb; e_diff = eb - ea; ma = ma >> e_diff;
        end

        // Add or subtract mantissas
        if (sa == sb) begin
            m_sum = {1'b0, ma} + {1'b0, mb};
            bf16_add = {sa, e_max + m_sum[8], m_sum[7:1]};
        end else if (ma >= mb) begin
            m_sum = {1'b0, ma} - {1'b0, mb};
            bf16_add = {sa, e_max, m_sum[7:1]};
        end else begin
            m_sum = {1'b0, mb} - {1'b0, ma};
            bf16_add = {sb, e_max, m_sum[7:1]};
        end

        // Zero handling
        if (a == 0) bf16_add = b;
        if (b == 0) bf16_add = a;
    endfunction

endmodule
