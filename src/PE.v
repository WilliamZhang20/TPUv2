module PE #(
    input  logic             clk,
    input  logic             rst,
    input  logic             clear,     // load new value
    input  logic [7:0]       a_in,      // FP8 E4M3
    input  logic [7:0]       b_in,
    output logic [7:0]       a_out,
    output logic [7:0]       b_out,
    output logic [15:0]      c_out       // BF16
);

    // ==================== FP8 decode & multiply (combinational) ====================
    logic        sign_p   = a_in[7] ^ b_in[7];
    logic [3:0]  exp_a    = a_in[6:3];
    logic [3:0]  exp_b    = b_in[6:3];
    logic [3:0]  mant_a   = (exp_a != 0) ? {1'b1, a_in[2:0]} : 4'd0;
    logic [3:0]  mant_b   = (exp_b != 0) ? {1'b1, b_in[2:0]} : 4'd0;
    logic [7:0]  mant_prod = mant_a * mant_b;                 // 0 or 64..225
    logic [8:0]  exp_sum   = exp_a + exp_b;
    logic        is_zero   = (mant_prod == 0) || (exp_a == 0) || (exp_b == 0);

    // ==================== FP8×FP8 → rounded BF16 (combinational) ====================
    logic [7:0]  bf_exp;
    logic [6:0]  bf_mant;
    logic [15:0] bf_prod;

    always_comb begin
        bf_exp  = 8'd0;
        bf_mant = 7'd0;

        if (!is_zero) begin
            logic [7:0] mant_tmp     = mant_prod[7] ? mant_prod : (mant_prod << 1);
            logic [8:0] exp_base     = mant_prod[7] ? 9'd120 : 9'd119;
            logic       round_up     = mant_tmp[0];
            logic [7:0] mant_rounded = mant_tmp[7:1] + round_up;

            bf_exp  = exp_sum + exp_base;
            if (mant_rounded[7]) begin
                bf_mant = mant_rounded[7:1];
                bf_exp  = bf_exp + 1;
            end else begin
                bf_mant = mant_rounded[6:0];
            end
        end
    end

    assign bf_prod = {sign_p, bf_exp, bf_mant};

    // ==================== Full combinational BF16 adder ====================
    function automatic logic [15:0] bf16_add(input [15:0] a, b);
        logic sa = a[15], sb = b[15];
        logic [7:0] ea = a[14:7], eb = b[14:7];
        logic [8:0] ma = (ea != 0) ? {2'b01, a[6:0]} : 9'd0;
        logic [8:0] mb = (eb != 0) ? {2'b01, b[6:0]} : 9'd0;
        logic [7:0] e_out;
        logic [9:0] m_sum;

        if (ea >= eb) begin
            e_out = ea;
            mb    = mb >> (ea - eb);
        end else begin
            e_out = eb;
            ma    = ma >> (eb - ea);
        end

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

    logic [15:0] sum = bf16_add(c_out, bf_prod);

    // ==================== Accumulator register (only thing clocked) ====================
    always_ff @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;

        if (rst)
            c_out <= 16'd0;
        else if (clear)
            c_out <= bf_prod;
        else
            c_out <= sum;           // sum is fully combinational → c_out valid SAME cycle
    end

endmodule
