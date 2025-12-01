module bf16_adder (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] sum
);
    logic a_sign = a[15];
    logic b_sign = b[15];
    logic [7:0] a_exp = a[14:7];
    logic [7:0] b_exp = b[14:7];
    logic [6:0] a_frac = a[6:0];
    logic [6:0] b_frac = b[6:0];

    logic [8:0] exp_diff;
    logic [8:0] big_exp, small_exp;
    logic [7:0] big_frac, small_frac;
    logic        swap;

    logic [9:0] big_mant = {1'b1, big_frac};
    logic [9:0] small_mant_aligned;
    logic [10:0] sum_mant;
    logic        sum_sign;

    always@(*) begin
        // Handle specials quickly
        if (a_exp == '1) sum = a;
        else if (b_exp == '1) sum = b;
        else if (a_exp == 0 && a_frac == 0) sum = b;
        else if (b_exp == 0 && b_frac == 0) sum = a;
        else begin
            swap = (a_exp < b_exp) || (a_exp == b_exp && a_frac < b_frac);
            big_exp   = swap ? b_exp : a_exp;
            small_exp = swap ? a_exp : b_exp;
            big_frac  = swap ? b_frac : a_frac;
            small_frac= swap ? a_frac : b_frac;

            exp_diff = big_exp - small_exp;

            small_mant_aligned = (exp_diff >= 10) ? 0 :
                                 {2'b1, small_frac, 1'b0} >> exp_diff;

            if (a_sign == b_sign) begin
                sum_mant = {1'b0, big_mant} + small_mant_aligned;
                sum_sign = a_sign;
            end else begin
                sum_mant = {1'b0, big_mant} - small_mant_aligned;
                sum_sign = swap ? b_sign : a_sign;
            end

            if (sum_mant[10]) begin
                sum = {sum_sign, big_exp+1, sum_mant[9:3]};
            end else if (sum_mant[9:0] == 0) begin
                sum = 16'b0;
            end else begin
                sum = {sum_sign, big_exp, sum_mant[9:3]};
            end
        end
    end
endmodule