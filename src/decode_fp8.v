module fp8_to_bf16_decoder #(
    parameter int FTZ_FP8 = 1,
    parameter int WIDTH = 8
)(
    input  logic [WIDTH-1:0] fp8,
    output logic             is_zero,
    output logic             is_inf,
    output logic             sign,
    output logic [7:0]       exp_out,
    output logic [7:0]            mant_hidden_out   // 1.mmmmmmm
);

    localparam int FP8_EXP_BITS = 4;
    localparam int FP8_MAN_BITS = 3;
    localparam int FP8_BIAS     = 7;
    localparam int BF16_EXP_BITS = 8;
    localparam int BF16_MAN_BITS = 7;
    localparam int BF16_BIAS     = 127;

    logic [FP8_EXP_BITS-1:0] e8;
    logic [FP8_MAN_BITS-1:0] m8;

    always_comb begin
        sign = fp8[WIDTH-1];
        e8   = fp8[WIDTH-2 -: FP8_EXP_BITS];
        m8   = fp8[FP8_MAN_BITS-1:0];

        is_zero = 1'b0;
        is_inf  = 1'b0;

        if (e8 == 0 && m8 == 0) begin
            is_zero = 1'b1;
            exp_out = 0;
            mant_hidden_out = 0;
        end else if (e8 == '1) begin
            is_inf  = 1'b1;
            exp_out = '1;
            mant_hidden_out = 0;
        end else if (e8 == 0) begin // subnormal
            if (FTZ_FP8 == 1) begin
                is_zero = 1'b1;
                exp_out = 0;
                mant_hidden_out = 0;
            end else begin
                exp_out = 0;
                mant_hidden_out = {1'b0, m8, 4'b0}; // align to top of BF16 fraction
            end
        end else begin // normal
            exp_out = e8 - FP8_BIAS + BF16_BIAS;
            mant_hidden_out = {1'b1, m8, 4'b0}; // hidden 1 + 3-bit mant shifted left by 4
        end
    end
endmodule