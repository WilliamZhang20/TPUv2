module mx_int_acc (
    input  wire        clk,
    input  wire        rst,
    input  wire        clear,
    input  wire [19:0] aligned_val,
    input  wire        prod_sign,
    output reg  signed [19:0] acc
);

    always @(posedge clk) begin
        if (rst) begin
            acc <= 0;
        end else if (clear) begin
            acc <= 0;
        end else begin
            acc <= prod_sign ? acc - aligned_val : acc + aligned_val;
        end
    end
endmodule

module mx_int_aligner (
    input  wire [7:0]  mant_prod,
    input  wire [7:0]  shift_amount,
    output wire [19:0] aligned_val     // 20-bit wide integer domain
);

    assign aligned_val = mant_prod << shift_amount;

endmodule
