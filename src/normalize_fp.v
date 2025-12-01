module int20_to_bf16 (
    input  wire signed [19:0] acc,
    output reg  [15:0] bf16
);

    localparam BF16_BIAS = 127;
    localparam INT_ACC_OFFSET = 24;

    wire sign = acc[19];
    wire [19:0] abs_val = sign ? -acc : acc;

    integer i;
    reg [4:0] lead;

    always @(*) begin
        if (abs_val == 0) begin
            bf16 = 16'h0000;
        end else begin
            // leading 1
            lead = 0;
            for (i = 19; i >= 0; i = i - 1)
                if (abs_val[i]) begin
                    lead = i;
                    break;
                end

            // compute expo
            wire [7:0] exp = lead + (BF16_BIAS - INT_ACC_OFFSET);

            // mantissa bits
            wire [6:0] mant = abs_val[lead-1 -: 7];

            bf16 = {sign, exp, mant};
        end
    end
endmodule
