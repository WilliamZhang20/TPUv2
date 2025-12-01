module int20_to_bf16 (
    input  wire signed [19:0] acc,
    output reg         [15:0] bf16
);
    localparam BF16_BIAS      = 127;
    localparam INT_ACC_OFFSET = 24;

    reg        sign;
    reg [19:0] mag;
    reg [4:0]  lz;
    reg [7:0]  exp;
    reg [6:0]  mant;

    always @(*) begin
        sign = acc[19];
        mag  = sign ? -acc : acc;

        if (mag == 0) begin
            bf16 = 16'h0000;
        end else begin
            // Count leading zeros (synthesis turns this into priority encoder)
            lz = 20;
            if (mag[19]) lz = 0;
            else if (mag[18]) lz = 1;
            else if (mag[17]) lz = 2;
            else if (mag[16]) lz = 3;
            else if (mag[15]) lz = 4;
            else if (mag[14]) lz = 5;
            else if (mag[13]) lz = 6;
            else if (mag[12]) lz = 7;
            else if (mag[11]) lz = 8;
            else if (mag[10]) lz = 9;
            else if (mag[9])  lz = 10;
            else if (mag[8])  lz = 11;
            else if (mag[7])  lz = 12;
            else if (mag[6])  lz = 13;
            else if (mag[5])  lz = 14;
            else if (mag[4])  lz = 15;
            else if (mag[3])  lz = 16;
            else if (mag[2])  lz = 17;
            else if (mag[1])  lz = 18;
            else if (mag[0])  lz = 19;

            // Position of leading 1 = 19 - lz
            exp  = (19 - lz) + (BF16_BIAS - INT_ACC_OFFSET);
            mant = lz < 13 ? mag << (lz + 1) : 7'd0;  // +1 to skip implicit bit

            bf16 = {sign, exp, mant};
        end
    end
endmodule