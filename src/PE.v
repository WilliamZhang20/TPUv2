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
    // 1. Correct FP8 E4M3 → BF16 conversion
    //    E4M3: bias = 7,  exp: 4 bits, mant: 3 bits (no implicit 1 for normalized!)
    //    BF16: bias = 127
    //    So new_exp = exp8 + (127 - 7) = exp8 + 120
    // =================================================================
    function automatic [15:0] fp8_e4m3_to_bf16(input [7:0] fp8);
        reg        sign;
        reg [3:0]  exp8;
        reg [2:0]  mant8;
        reg [7:0]  exp16;
        reg [6:0]  mant16;
        reg        is_zero_or_subnormal;
    begin
        sign   = fp8[7];
        exp8   = fp8[6:3];
        mant8  = fp8[2:0];

        is_zero_or_subnormal = (exp8 == 4'd0);

        if (is_zero_or_subnormal) begin
            // Subnormals: treat as zero (common in low-precision training)
            // Or you can denormalize: mant16 = mant8 << 4; exp16 = 127 - 6 = 121
            // Here we flush subnormals to zero for simplicity and area
            exp16  = 8'd0;
            mant16 = 7'd0;
        end else begin
            // Normalized: implicit leading 1
            exp16  = 8'd120 + exp8;        // bias adjust: 127 - 7 = 120
            mant16 = {mant8, 4'b0000};     // 3 → 7 bits: pad 4 zeros (not 5!)
        end

        fp8_e4m3_to_bf16 = {sign, exp16, mant16};
    end
    endfunction

    // =================================================================
    // 2. Correct BF16 × BF16 → BF16 multiplier (area-optimized)
    // =================================================================
    function automatic [15:0] bf16_mul(input [15:0] a, input [15:0] b);
        reg        sa, sb, sp;
        reg [7:0]  ea, eb, ep;
        reg [7:0]  ma, mb;        // 7-bit mantissa (implicit 1 not stored)
        reg [15:0] mp;            // 8 × 8 → 16 bits max
        reg        sticky;
        reg [7:0]  ep_final;
        reg [6:0]  mp_final;
    begin
        sa = a[15]; sb = b[15]; sp = sa ^ sb;
        ea = a[14:7]; ma = a[6:0];
        eb = b[14:7]; mb = b[6:0];

        // Special cases
        if (ea == 0 || eb == 0) begin
            bf16_mul = {sp, 15'd0};
        end else if (ea == 8'hFF || eb == 8'hFF) begin
            bf16_mul = {sp, 8'hFF, 7'd0};  // Inf
        end else begin
            ep = ea + eb - 8'd127;
            mp = {1'b1, ma} * {1'b1, mb};  // 8x8 → 16 bits

            // Round-to-nearest-even
            if (mp[15]) begin
                // 1x.mmmmmmmm...
                sticky     = |mp[7:0];
                mp_final   = mp[14:8] + (mp[7] & (mp[8] | sticky));
                ep_final   = ep + 1'b1;
            end else begin
                // 01.mmmmmmm...
                sticky     = |mp[6:0];
                mp_final   = mp[13:7] + (mp[6] & (mp[7] | sticky));
                ep_final   = ep;
            end

            // Overflow check
            if (ep_final >= 255) begin
                bf16_mul = {sp, 8'hFF, 7'd0};
            end else begin
                bf16_mul = {sp, ep_final, mp_final};
            end
        end
    end
    endfunction

    // =================================================================
    // 3. Correct BF16 + BF16 → BF16 adder (for accumulation)
    // =================================================================
    function automatic [15:0] bf16_add(input [15:0] a, input [15:0] b);
        reg        sa, sb;
        reg [7:0]  ea, eb, e_diff, e_max;
        reg [8:0]  mant_a, mant_b;  // 1.8 format (for alignment + guard)
        reg [9:0]  mant_sum;        // extra bit for carry
        reg [7:0]  e_result;
        reg [6:0]  m_result;
        reg        sign_result;
    begin
        sa = a[15]; ea = a[14:7];
        sb = b[15]; eb = b[14:7];

        // Handle zero
        if (ea == 0) mant_a = 9'd0;
        else         mant_a = {1'b1, a[6:0], 1'b0};  // implicit bit + guard

        if (eb == 0) mant_b = 9'd0;
        else         mant_b = {1'b1, b[6:0], 1'b0};

        if (ea > eb) begin
            e_max  = ea;
            e_diff = ea - eb;
            mant_b = mant_b >> e_diff;
        end else begin
            e_max  = eb;
            e_diff = eb - ea;
            mant_a = mant_a >> e_diff;
        end

        if (sa == sb) begin
            mant_sum = {1'b0, mant_a} + {1'b0, mant_b};
            sign_result = sa;
        end else if (mant_a >= mant_b) begin
            mant_sum = {1'b0, mant_a} - {1'b0, mant_b};
            sign_result = sa;
        end else begin
            mant_sum = {1'b0, mant_b} - {1'b0, mant_a};
            sign_result = sb;
        end

        // Normalize
        if (mant_sum[9]) begin
            m_result = mant_sum[9:3] + mant_sum[2];  // round up if needed
            e_result = e_max + 1;
        end else if (mant_sum[8]) begin
            m_result = mant_sum[8:2] + mant_sum[1];
            e_result = e_max;
        end else begin
            // Should not happen for normalized inputs, but safe
            m_result = 7'd0;
            e_result = 8'd0;
        end

        // Overflow → Inf
        if (e_result >= 255) begin
            bf16_add = {sign_result, 8'hFF, 7'd0};
        end else begin
            bf16_add = {sign_result, e_result, m_result};
        end
    end
    endfunction

    // =================================================================
    // Pipeline registers and MAC
    // =================================================================
    wire [15:0] a_bf16 = fp8_e4m3_to_bf16(a_in);
    wire [15:0] b_bf16 = fp8_e4m3_to_bf16(b_in);
    wire [15:0] product = bf16_mul(a_bf16, b_bf16);

    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;

        if (rst) begin
            c_out <= 16'd0;
        end else if (clear) begin
            c_out <= product;
        end else begin
            c_out <= bf16_add(c_out, product);
        end
    end

endmodule