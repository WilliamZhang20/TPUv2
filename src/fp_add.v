module bf16_adder_simple (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] sum
);

    wire a_sign = a[15];
    wire b_sign = b[15];
    wire [7:0] a_exp = a[14:7];
    wire [7:0] b_exp = b[14:7];
    wire [6:0] a_frac = a[6:0];
    wire [6:0] b_frac = b[6:0];

    // Temporary combinational regs
    reg        swap;
    reg [7:0]  big_exp, small_exp;
    reg [6:0]  big_frac, small_frac;
    reg        big_sign, small_sign;

    reg [7:0]  exp_diff;
    reg [9:0]  big_mant, small_mant_aligned;

    reg [10:0] result_mant;
    reg        result_sign;

    always @(*) begin

        // Handle special cases
        if (a_exp == 8'hFF) begin
            sum = a;
        end
        else if (b_exp == 8'hFF) begin
            sum = b;
        end
        else if ((a_exp == 0) && (a_frac == 0)) begin
            sum = b;
        end
        else if ((b_exp == 0) && (b_frac == 0)) begin
            sum = a;
        end
        else begin
            // Sorting
            swap = (a_exp < b_exp) ||
                   ((a_exp == b_exp) && (a_frac < b_frac));

            big_exp    = swap ? b_exp    : a_exp;
            small_exp  = swap ? a_exp    : b_exp;
            big_frac   = swap ? b_frac   : a_frac;
            small_frac = swap ? a_frac   : b_frac;

            big_sign   = swap ? b_sign   : a_sign;
            small_sign = swap ? a_sign   : b_sign;

            exp_diff = big_exp - small_exp;

            big_mant  = {1'b1, big_frac, 2'b00};

            if (exp_diff >= 10)
                small_mant_aligned = 10'b0;
            else
                small_mant_aligned = {1'b1, small_frac, 2'b00} >> exp_diff;

            // Addition or subtraction
            if (big_sign == small_sign) begin
                result_mant = {1'b0, big_mant} + small_mant_aligned;
                result_sign = big_sign;
            end else begin
                result_mant = {1'b0, big_mant} - small_mant_aligned;
                result_sign = big_sign;
            end

            // Normalize + repack bfloat16
            if (result_mant == 0) begin
                sum = 16'h0000;
            end
            else if (result_mant[10]) begin
                sum = {result_sign, big_exp + 8'd1, result_mant[9:3]};
            end else begin
                sum = {result_sign, big_exp, result_mant[9:3]};
            end
        end
    end
endmodule
