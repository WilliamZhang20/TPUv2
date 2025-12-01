module int20_to_bf16 (
    input  wire signed [19:0] acc,
    output reg  [15:0] bf16
);

    // Constants
    localparam BF16_BIAS      = 127;
    localparam INT_ACC_OFFSET = 24;

    // Internal variables
    integer i;
    reg [4:0] lead;
    reg sign;
    reg [19:0] abs_val;
    reg [7:0] exp;
    reg [6:0] mant;

    always @(*) begin
        // Extract sign and absolute value
        sign    = acc[19];
        abs_val = sign ? -acc : acc;

        if (abs_val == 0) begin
            bf16 = 16'h0000;  // Zero case
        end else begin
            // Find leading 1
            lead = 0;
            lead_loop: for (i = 19; i >= 0; i = i - 1) begin
                if (abs_val[i]) begin
                    lead = i[4:0];
                    disable lead_loop;  // break the loop
                end
            end

            // Compute exponent
            exp = lead + (BF16_BIAS - INT_ACC_OFFSET);

            // Compute mantissa
            if (lead >= 7)
                mant = abs_val[lead-1 -: 7];
            else
                mant = abs_val[6:0] << (7 - lead);

            // Assemble BF16 result
            bf16 = {sign, exp, mant};
        end
    end

endmodule
