module PE #(
    parameter WIDTH = 8
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    clear,
    input  wire signed [WIDTH-1:0] a_in,   // FP8 E4M3
    input  wire signed [WIDTH-1:0] b_in,   // FP8 E4M3
    output reg  signed [WIDTH-1:0] a_out,
    output reg  signed [WIDTH-1:0] b_out,
    output reg  signed [15:0]      c_out   // BF16 accumulator
);

    // =================================================================
    // 1. FP8 (E4M3) → BF16 correct conversion (bias 7 → 127 ⇒ +120)
    // =================================================================
    function automatic [15:0] fp8_to_bf16(input [7:0] fp8);
        reg        sign;
        reg [3:0]  exp8;
        reg [2:0]  mant8;
        reg [7:0]  exp16;
        reg [6:0]  mant16;
    begin
        sign  = fp8[7];
        exp8  = fp8[6:3];
        mant8 = fp8[2:0];

        if (exp8 == 4'd0) begin
            // zero or subnormal → BF16 subnormal or zero
            mant16 = {mant8, 4'b0};                 // shift into BF16 mantissa field
            exp16  = 8'd0;
            if (mant8 == 0) mant16 = 7'd0;
        end else if (exp8 == 4'd15) begin
            // Inf or NaN
            exp16  = 8'd255;
            mant16 = (mant8 == 0) ? 7'd0 : {1'b1, mant8[2:0], 3'b0};
        end else begin
            // normal number
            exp16  = exp8 + 8'd120;                 // bias adjustment
            mant16 = {1'b1, mant8} << 3;            // implicit bit + shift left 3
        end
        fp8_to_bf16 = {sign, exp16, mant16};
    end
    endfunction

    // =================================================================
    // 2. BF16 × BF16 → BF16 multiplier (no extra bits, no FP32)
    // =================================================================
    function automatic [15:0] bf16_mul(input [15:0] a, input [15:0] b);
        reg        sa, sb, sp;
        reg [7:0]  ea, eb;
        reg [7:0]  ep;
        reg [6:0]  ma, mb;
        reg [13:0] mp;
        reg [7:0]  ep_final;
        reg [6:0]  mp_final;
        reg        round_bit;
    begin
        sa = a[15]; sb = b[15];
        ea = a[14:7];  ma = a[6:0];
        eb = b[14:7];  mb = b[6:0];

        sp = sa ^ sb;

        // Special cases
        if (ea == 0 || eb == 0) begin
            bf16_mul = {sp, 15'd0};                     // zero
        end else if (ea == 255 || eb == 255) begin
            bf16_mul = {sp, 8'd255, 7'd0};              // Inf (quiet NaN ignored for area)
        end else begin
            // Normal path
            ep = ea + eb - 127;                         // exponent sum (biased)
            mp = {1'b1, ma} * {1'b1, mb};               // 8 × 8 → 14-bit mantissa product

            // Normalize (at most 1-bit shift)
            if (mp[13]) begin
                mp_final   = mp[12:6];                  // take bits 12:6 → 7-bit mantissa
                round_bit  = mp[5];
                ep_final   = ep + 1;
            end else begin
                mp_final   = mp[11:5];
                round_bit  = mp[4];
                ep_final   = ep;
            end

            // Round-to-nearest-even (tie-to-even)
            if (round_bit && (mp_final[0] || |mp[3:0]))
                mp_final = mp_final + 1'b1;

            // Overflow → Inf
            if (ep_final == 8'hFF)
                bf16_mul = {sp, 8'd255, 7'd0};
            else
                bf16_mul = {sp, ep_final, mp_final};
        end
    end
    endfunction

    // =================================================================
    // 3. BF16 + BF16 → BF16 adder (block floating-point style)
    // =================================================================
    function automatic [15:0] bf16_add(input [15:0] a, input [15:0] b);
        reg        sa, sb;
        reg [7:0]  ea, eb;
        reg [7:0]  e_max;
        reg [8:0]  mant_a, mant_b;   // 1.7 format (implicit bit included)
        reg [8:0]  mant_sum;
        reg [7:0]  e_result;
        reg [6:0]  m_result;
    begin
        sa = a[15]; ea = a[14:7]; mant_a = {1'b1, a[6:0], 1'b0};
        sb = b[15]; eb = b[14:7]; mant_b = {1'b1, b[6:0], 1'b0};

        if (ea > eb) begin
            e_max    = ea;
            mant_b   = mant_b >> (ea - eb);
        end else begin
            e_max    = eb;
            mant_a   = mant_a >> (eb - ea);
        end

        if (sa == sb)
            mant_sum = mant_a + mant_b;
        else
            mant_sum = (sa ? mant_b - mant_a : mant_a - mant_b);

        // Normalize
        if (mant_sum[8]) begin
            m_result  = mant_sum[8:2];
            e_result  = e_max + 1;
        end else begin
            m_result  = mant_sum[7:1];
            e_result  = e_max;
        end

        // Simple overflow → Inf
        bf16_add = (e_result >= 255) ? {sa, 8'd255, 7'd0} : {sa, e_result, m_result};
    end
    endfunction

    // =================================================================
    // Pipeline and accumulation
    // =================================================================
    wire [15:0] a_bf16 = fp8_to_bf16(a_in);
    wire [15:0] b_bf16 = fp8_to_bf16(b_in);
    wire [15:0] product = bf16_mul(a_bf16, b_bf16);

    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;

        if (rst)
            c_out <= 16'd0;
        else if (clear)
            c_out <= product;
        else
            c_out <= bf16_add(c_out, product);
    end

endmodule