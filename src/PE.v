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
    wire denorm_a = ~|exp_a;
    wire denorm_b = ~|exp_b;

    // Combine mantissa selection into single operation
    wire [3:0] mant_a = {~denorm_a, a_in[2:0]};
    wire [3:0] mant_b = {~denorm_b, b_in[2:0]};

    // ----------------------- Multiply & align -----------------------
    wire prod_sign = sign_a ^ sign_b;
    wire [7:0] mant_prod = mant_a * mant_b;   // 4×4 = 8-bit product

    localparam signed BIAS = -7;
    localparam FRAC_BITS = 8;
    
    // Simplified exponent calculation (reuse adder)
    wire signed [4:0] exp_a_add = denorm_a ? 5'sd1 : {1'b0, exp_a};
    wire signed [4:0] exp_b_add = denorm_b ? 5'sd1 : {1'b0, exp_b};
    wire signed [5:0] exp_add_only = exp_a_add + exp_b_add;
    wire signed [6:0] exp_sum = {1'b0, exp_add_only} - 7'sd14;
    wire signed [6:0] shift_right = 7'sd6 - FRAC_BITS - exp_sum;
    
    // Simplified shifter with clamping
    reg [17:0] aligned_prod;
    always @(*) begin
        if (shift_right[6])  // Negative = left shift
            aligned_prod = (shift_right < -7'sd10) ? 18'h3FFFF : ({10'd0, mant_prod} << (-shift_right[3:0]));
        else  // Positive = right shift
            aligned_prod = (shift_right > 7'sd17) ? 18'd0 : ({10'd0, mant_prod} >> shift_right[4:0]);
    end

    // ----------------------- Accumulator (2's complement) -----------------------
    reg signed [17:0] acc;

    // Compute next accumulator value (combinational)
    wire signed [17:0] abs_aligned = aligned_prod;
    wire signed [17:0] acc_next = prod_sign ? (acc - abs_aligned) : (acc + abs_aligned);

    // Register accumulator
    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;
        acc <= (rst | clear) ? 18'sd0 : acc_next;
    end

    // ----------------------- INT18 → BF16 (combinational) -----------------------
    wire [15:0] bf16_c;
    int18_to_bf16_lzd #(.FRAC_BITS(FRAC_BITS)) convert (
        .acc(acc_next), 
        .bf16(bf16_c)
    );

    assign c_out = bf16_c;


endmodule
