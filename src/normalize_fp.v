module bf16_normalizer (
    input  logic        sign,
    input  logic [8:0]  exp_in,
    input  logic [15:0] mant_raw,
    output logic [15:0] bf16_result
);

    logic [7:0]  mant_norm;
    logic [7:0]  exp_final;

    always @(*) begin
        if (mant_raw == 0) begin
            bf16_result = {sign, 15'b0};
        end else if (mant_raw[15]) begin // >= 2.0 → shift right
            mant_norm = mant_raw[15:8] + mant_raw[7];      // round-to-nearest (add guard bit)
            exp_final = exp_in + 1;
        end else begin
            mant_norm = mant_raw[14:7] + mant_raw[6];      // round-to-nearest
            exp_final = exp_in;
        end

        // Overflow → Inf
        if (exp_final[8] || exp_final[7:0] == 8'hFF) begin
            bf16_result = {sign, 8'hFF, 7'b0};
        end else if (exp_final[7:0] == 0) begin
            bf16_result = {sign, 15'b0};                   // underflow → zero
        end else begin
            bf16_result = {sign, exp_final[7:0], mant_norm[6:0]};
        end
    end
endmodule