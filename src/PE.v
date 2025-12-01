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

    // -----------------------
    // Decode FP8 E4M3
    // -----------------------
    wire sign_a = a_in[7];
    wire sign_b = b_in[7];

    wire [3:0] exp_a  = a_in[6:3];
    wire [3:0] exp_b  = b_in[6:3];

    wire       denorm_a = (exp_a == 0);
    wire       denorm_b = (exp_b == 0);

    wire [3:0] mant_a = denorm_a ? {1'b0, a_in[2:0]} :
                                   {1'b1, a_in[2:0]};

    wire [3:0] mant_b = denorm_b ? {1'b0, b_in[2:0]} :
                                   {1'b1, b_in[2:0]};

    // -----------------------
    // Multiply + Align
    // -----------------------
    wire prod_sign = sign_a ^ sign_b;
    wire [7:0] mant_prod = mant_a * mant_b;

    localparam FP8_BIAS = 7;
    localparam INT_ACC_OFFSET = 18;

    wire [7:0] shift_amount =
        (exp_a + exp_b - FP8_BIAS + INT_ACC_OFFSET);

    wire [17:0] aligned_prod =
        $unsigned(mant_prod) << shift_amount;

    // -----------------------
    // Accumulator (only registered stage)
    // -----------------------
    reg signed [17:0] acc;

    always @(posedge clk) begin
        if (rst || clear)
            acc <= 0;
        else
            acc <= prod_sign ? acc - aligned_prod :
                               acc + aligned_prod;
    end

    // -----------------------
    // INT18 → BF16 conversion (combinational)
    // -----------------------
    wire [15:0] bf16_c;

    int18_to_bf16_lzd convert(
        .acc(acc),
        .bf16(bf16_c)
    );

    // -----------------------
    // Output register = 1 cycle latency
    // -----------------------
    always @(posedge clk) begin
        if (rst)
            c_out <= 16'h0000;
        else
            c_out <= bf16_c;
    end

endmodule