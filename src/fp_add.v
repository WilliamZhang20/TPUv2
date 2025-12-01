module bf16_adder (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] sum
);

    logic a_sign, b_sign;
    logic [7:0] a_exp, b_exp;
    logic [6:0] a_frac, b_frac;

    // unpacked fields
    always @(*) begin
        a_sign = a[15];
        b_sign = b[15];
        a_exp  = a[14:7];
        b_exp  = b[14:7];
        a_frac = a[6:0];
        b_frac = b[6:0];
    end

    always @(*) begin
        // default to avoid latches
        sum = a;  // safe default

        // special cases
        if (a_exp == 8'hFF) begin
            sum = a;                                          // inf or NaN
        end else if (b_exp == 8'hFF) begin
            sum = b;
        end else if (a_exp == 0 && a_frac == 0) begin
            sum = b;
        end else if (b_exp == 0 && b_frac == 0) begin
            sum = a;
        end else begin
            // normal add
            logic swap;
            logic [7:0] big_exp, small_exp;
            logic [6:0] big_frac, small_frac;
            logic [8:0] exp_diff;
            logic [9:0] big_mant;
            logic [9:0] small_mant_aligned;
            logic [10:0] sum_mant;
            logic        sum_sign;

            swap       = (a_exp < b_exp) || ((a_exp == b_exp) && (a_frac < b_frac));
            big_exp    = swap ? b_exp  : a_exp;
            small_exp  = swap ? a_exp  : b_exp;
            big_frac   = swap ? b_frac : a_frac;
            small_frac = swap ? a_frac : b_frac;

            exp_diff = big_exp - small_exp;

            big_mant = {1'b1, big_frac};  // hidden bit

            if (exp_diff >= 10) begin
                small_mant_aligned = 0;
            end else begin
                small_mant_aligned = {1'b1, small_frac, 2'b00} >> exp_diff;  // +2 bits for guard+round
            end

            if (a_sign == b_sign) begin
                sum_mant = {1'b0, big_mant} + small_mant_aligned;
                sum_sign = a_sign;
            end else begin
                sum_mant = {1'b0, big_mant} - small_mant_aligned;
                sum_sign = swap ? b_sign : a_sign;
            end

            if (sum_mant[10]) begin
                sum = {sum_sign, big_exp + 1'd1, sum_mant[9:3]};
            end else if (sum_mant[9:0] == 0) begin
                sum = 16'h0000;
            end else begin
                sum = {sum_sign, big_exp, sum_mant[9:3]};
            end
        end
    end

endmodule