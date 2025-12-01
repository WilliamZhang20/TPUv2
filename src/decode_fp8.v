module mxfp8_e4m3_decoder (
    input  wire [7:0] fp8,
    output wire       sign,
    output wire [3:0] exp,
    output wire [3:0] mant  // hidden-bit included: [3] is implicit 1
);

    assign sign = fp8[7];
    assign exp  = fp8[6:3];

    wire denorm = (exp == 4'b0000);

    assign mant = denorm ? {1'b0, fp8[2:0]}  // no hidden 1
                         : {1'b1, fp8[2:0]}; // hidden 1 for normal
endmodule
