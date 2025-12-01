module bf16_normalizer (
    input  logic        sign,
    input  logic [8:0]  exp_in,
    input  logic [15:0] mant_raw,
    output logic [15:0] bf16_result
);

    always @(*) begin
        // default assignment — crucial!
        bf16_result = {sign, 15'b0};

        if (mant_raw == 0) begin
            bf16_result = {sign, 15'b0};
        end
        else if (mant_raw[15]) begin
            // product >= 2.0 → shift right 1, round using bit[7]
            logic [7:0] mant_rnd = mant_raw[15:8] + mant_raw[7];
            logic [7:0] exp_adj  = exp_in + 1'b1;
            if (exp_adj[8] || exp_adj[7:0] == 8'hFF)
                bf16_result = {sign, 8'hFF, 7'b0};     // overflow → inf
            else
                bf16_result = {sign, exp_adj[7:0], mant_rnd[6:0]};
        end
        else begin
            // normal case < 2.0
            logic [7:0] mant_rnd = mant_raw[14:7] + mant_raw[6];
            logic [7:0] exp_adj  = exp_in;
            if (exp_adj == 0)
                bf16_result = {sign, 15'b0};          // underflow → zero
            else
                bf16_result = {sign, exp_adj[7:0], mant_rnd[6:0]};
        end
    end

endmodule