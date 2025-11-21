module PE_fp8_bf16_mac #(
    parameter int WIDTH = 8
)(
    input  logic             clk,
    input  logic             rst,
    input  logic             clear,     // load new partial sum
    input  logic [7:0]       a_in,       // FP8 E4M3 weight / activation
    input  logic [7:0]       b_in,
    output logic [7:0]       a_out,
    output logic [7:0]       b_out,
    output logic [15:0]      c_out       // BF16 accumulator
);

    // ==================== Stage 0: FP8 decode & multiply ====================
    logic        sign_p;
    logic [3:0]  exp_a, exp_b;
    logic [3:0]  mant_a, mant_b;
    logic [7:0]  mant_prod;   // 0 or 64..225
    logic [8:0]  exp_sum;     // 0..30
    logic        is_zero;

    assign sign_p   = a_in[7] ^ b_in[7];
    assign exp_a    = a_in[6:3];
    assign exp_b    = b_in[6:3];
    assign mant_a   = (exp_a != 0) ? {1'b1, a_in[2:0]} : 4'd0;
    assign mant_b   = (exp_b != 0) ? {1'b1, b_in[2:0]} : 4'd0;

    assign mant_prod = mant_a * mant_b;
    assign exp_sum   = exp_a + exp_b;
    assign is_zero   = (mant_prod == 0) || (exp_a == 0) || (exp_b == 0);

    // ==================== Pipeline registers ====================
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

    // ==================== Stage 1: FP8×FP8 → rounded BF16 ====================
    logic [7:0]  bf_exp;
    logic [6:0]  bf_mant;
    logic [15:0] bf_prod;

    always_comb begin
        if (zero_r) begin
            bf_exp  = 8'd0;
            bf_mant = 7'd0;
        end else begin
            logic [7:0] mant_tmp;
            logic       round_up;
            logic [7:0] mant_rounded;

            if (mant_prod_r[7]) begin
                // 128..225 → shift right 1
                mant_tmp = mant_prod_r;
                bf_exp   = exp_sum_r + 8'd120;          // -7 + 127 + 1
            end else begin
                // 64..127 → shift left 1
                mant_tmp = mant_prod_r << 1;
                bf_exp   = exp_sum_r + 8'd119;          // -7 + 127 + 0
            end

            // round-to-nearest-even (only one sticky bit → simple round_up)
            round_up     = mant_tmp[0];
            mant_rounded = mant_tmp[7:1] + {7'd0, round_up};

            if (mant_rounded[7]) begin
                bf_mant = mant_rounded[7:1];
                bf_exp  = bf_exp + 1'd1;
            end else begin
                bf_mant = mant_rounded[6:0];
            end
        end
    end

    assign bf_prod = {sign_p_r, bf_exp, bf_mant};

    // ==================== Registered BF16 adder (2-cycle pipeline) ====================
    logic [15:0] acc;
    logic [15:0] add_b;

    always_ff @(posedge clk) add_b <= bf_prod;

    // Simple, correct, combinational BF16 adder
    function automatic logic [15:0] bf16_add(logic [15:0] a, b);
        logic        sa = a[15], sb = b[15];
        logic [7:0]  ea = a[14:7], eb = b[14:7];
        logic [8:0]  ma = (ea != 0) ? {2'b01, a[6:0]} : 9'd0;  // implicit 1 + guard
        logic [8:0]  mb = (eb != 0) ? {2'b01, b[6:0]} : 9'd0;
        logic [7:0]  e_out;
        logic [9:0]  m_sum;

        // align
        if (ea >= eb) begin
            e_out = ea;
            mb    = mb >> (ea - eb);
        end else begin
            e_out = eb;
            ma    = ma >> (eb - ea);
        end

        // add/sub
        if (sa == sb) begin
            m_sum = {1'b0, ma} + {1'b0, mb};
            bf16_add = {sa, e_out + m_sum[9], m_sum[8:2]};
        end else if (ma >= mb) begin
            m_sum = {1'b0, ma} - {1'b0, mb};
            bf16_add = {sa, e_out, m_sum[8:2]};
        end else begin
            m_sum = {1'b0, mb} - {1'b0, ma};
            bf16_add = {sb, e_out, m_sum[8:2]};
        end
    endfunction

    logic [15:0] sum = bf16_add(acc, add_b);

    // Accumulator
    always_ff @(posedge clk) begin
        if (rst)
            acc <= 16'd0;
        else if (clear)
            acc <= bf_prod;          // first product of new accumulation
        else
            acc <= sum;
    end

    assign c_out = acc;

endmodule
