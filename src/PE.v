module PE (
    input  wire        clk,
    input  wire        rst,
    input  wire        clear,
    input  wire [7:0]  a_in,
    input  wire [7:0]  b_in,
    output reg  [7:0]  a_out,
    output reg  [7:0]  b_out,
    output wire [15:0] c_out
);

    // ----------------------- FP8 E4M3 decode -----------------------
    wire sign_a = a_in[7];
    wire sign_b = b_in[7];
    wire [3:0] exp_a = a_in[6:3];
    wire [3:0] exp_b = b_in[6:3];
    wire denorm_a = (exp_a == 0);
    wire denorm_b = (exp_b == 0);

    wire [3:0] mant_a = denorm_a ? {1'b0, a_in[2:0]} : {1'b1, a_in[2:0]};
    wire [3:0] mant_b = denorm_b ? {1'b0, b_in[2:0]} : {1'b1, b_in[2:0]};

    // ----------------------- Multiply & align -----------------------
    wire prod_sign = sign_a ^ sign_b;
    wire [7:0] mant_prod = mant_a * mant_b;   // 4×4 = 8-bit product

    localparam FP8_BIAS = 7;
    localparam FRAC_BITS = 8;  // Q10.8 format: 1 sign + 9 int + 8 frac = 18 bits
    
    // Compute unbiased exponent of product
    // For denormals, use effective exponent of 1-bias = 1-7 = -6
    wire signed [5:0] exp_prod = (denorm_a ? (-6) : ($signed({2'b0, exp_a}) - FP8_BIAS)) + 
                                  (denorm_b ? (-6) : ($signed({2'b0, exp_b}) - FP8_BIAS));
    
    // Product mantissa has 6 fractional bits (3+3)
    // To align to FRAC_BITS position, we need to shift by (FRAC_BITS - 6) + exp_prod
    // Negative shift_right means shift left
    wire signed [5:0] shift_right = 6'd6 - FRAC_BITS - exp_prod;
    
    reg [17:0] aligned_prod;
    
    // Handle shift with saturation
    always @(*) begin
        if (shift_right >= 18) begin
            aligned_prod = 18'd0;  // Underflow
        end else if (shift_right <= -10) begin
            aligned_prod = 18'h3FFFF;  // Overflow (saturate to max)
        end else if (shift_right >= 0) begin
            aligned_prod = {10'd0, mant_prod} >> shift_right;
        end else begin
            // shift_right is negative, so negate it for left shift
            aligned_prod = {10'd0, mant_prod} << (-shift_right);
        end
    end

    // ----------------------- Accumulator (2's complement) -----------------------
    reg signed [17:0] acc;
    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;
        // $display("a_in %d, b_in %d, c_out %d", a_in, b_in, c_out);
        if (rst || clear)
            acc <= 18'sd0;
        else begin
            if (prod_sign)
                acc <= acc - $signed(aligned_prod);
            else
                acc <= acc + $signed(aligned_prod);
        end
    end

    // ----------------------- INT18 → BF16 (combinational) -----------------------
    wire [15:0] bf16_c;
    int18_to_bf16_lzd #(.FRAC_BITS(FRAC_BITS)) convert (
        .acc(acc), 
        .bf16(bf16_c)
    );

    assign c_out = bf16_c;

endmodule