module PE #(
    parameter WIDTH = 24      // efficient accumulator size
)(
    input  wire         clk,
    input  wire         rst,
    input  wire         clear,

    input  wire [7:0]   a_in,     // FP8 E4M3 (subnormals flushed)
    input  wire [7:0]   b_in,

    output reg  [7:0]   a_out,
    output reg  [7:0]   b_out,
    output reg  [15:0]  c_out     // BF16 output after convert
);

    // --------------------------------------------------------------------
    // 1. Ultra-light FP8 Decode
    // Subnormals → zero
    // Mantissa = 1.xxx (4-bit int), Exponent = exp - 7
    // --------------------------------------------------------------------
    wire sign_a = a_in[7];
    wire sign_b = b_in[7];
    wire sign_p = sign_a ^ sign_b;

    wire [3:0] exp_a  = a_in[6:3];
    wire [3:0] exp_b  = b_in[6:3];

    wire [3:0] mant_a = (exp_a == 4'd0) ? 4'd0 : {1'b1, a_in[2:0]};
    wire [3:0] mant_b = (exp_b == 4'd0) ? 4'd0 : {1'b1, b_in[2:0]};

    wire signed [5:0] e_a = $signed({1'b0, exp_a}) - 6'd7;
    wire signed [5:0] e_b = $signed({1'b0, exp_b}) - 6'd7;

    // --------------------------------------------------------------------
    // 2. INT multiply + shift exponent sum
    // --------------------------------------------------------------------
    wire [7:0] mant_prod = mant_a * mant_b; // 8-bit

    wire signed [6:0] shift_amt = e_a + e_b;

    // Shift in a bounded range (-7..+8 typical)
    wire signed [23:0] prod_shifted = 
        (shift_amt >= 0) ? 
            ({{16{1'b0}}, mant_prod} << shift_amt) :
            ({{16{1'b0}}, mant_prod} >> (-shift_amt));

    wire signed [23:0] prod = sign_p ? -prod_shifted : prod_shifted;

    // --------------------------------------------------------------------
    // 3. INT accumulator
    // --------------------------------------------------------------------
    reg signed [23:0] acc;

    // Convert INT24 → BF16 (approx, efficient)
    function automatic [15:0] int24_to_bf16(input signed [23:0] x);
        reg sign;
        reg [23:0] mag;
        reg [7:0] exponent;
        reg [6:0] mant;
    begin
        if (x == 0) begin
            int24_to_bf16 = 16'h0000;
        end else begin
            sign = x[23];
            mag = sign ? -x : x;

            // Normalize by finding highest bit
            // Since ACC_WIDTH small, simple if-chain
            if (mag[23]) begin exponent = 127+23; mant = mag[22:16]; end
            else if (mag[22]) begin exponent = 127+22; mant = mag[21:15]; end
            else if (mag[21]) begin exponent = 127+21; mant = mag[20:14]; end
            else if (mag[20]) begin exponent = 127+20; mant = mag[19:13]; end
            else if (mag[19]) begin exponent = 127+19; mant = mag[18:12]; end
            else if (mag[18]) begin exponent = 127+18; mant = mag[17:11]; end
            else if (mag[17]) begin exponent = 127+17; mant = mag[16:10]; end
            else if (mag[16]) begin exponent = 127+16; mant = mag[15:9]; end
            else if (mag[15]) begin exponent = 127+15; mant = mag[14:8]; end
            else if (mag[14]) begin exponent = 127+14; mant = mag[13:7]; end
            else if (mag[13]) begin exponent = 127+13; mant = mag[12:6]; end
            else begin exponent = 0; mant = 0; end

            int24_to_bf16 = {sign, exponent, mant};
        end
    end
    endfunction

    // --------------------------------------------------------------------
    // 4. Pipeline
    // --------------------------------------------------------------------
    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;

        if (rst)
            acc <= 24'd0;
        else if (clear)
            acc <= prod;
        else
            acc <= acc + prod;

        // Convert accumulator to BF16 each cycle (or only at readout)
        c_out <= int24_to_bf16(acc);
    end

endmodule

