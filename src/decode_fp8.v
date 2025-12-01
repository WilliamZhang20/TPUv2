module mx_decoder #(
    parameter int ELEM_WIDTH = 6
)(
    input  wire [ELEM_WIDTH-1:0] mx_elem,
    output wire sign,
    output wire [ELEM_WIDTH-2:0] mantissa
);
    assign sign     = mx_elem[ELEM_WIDTH-1];
    assign mantissa = mx_elem[ELEM_WIDTH-2:0];
endmodule


// ----------------------------------------------------------------------------
// Block Exponent Encoder (converts exponent to control bits)
// ----------------------------------------------------------------------------
module block_exp_encoder (
    input  wire [7:0] exponent,
    input  wire new_block,
    input  wire [5:0] elem_data,
    output wire [7:0] encoded_output
);
    wire [1:0] ctrl_bits = new_block ? 2'b01 : 2'b00;
    assign encoded_output = {ctrl_bits, elem_data};
endmodule


// ----------------------------------------------------------------------------
// Block Exponent Decoder (extract control + data)
// ----------------------------------------------------------------------------
module block_exp_decoder (
    input  wire [7:0] encoded_input,
    output wire [1:0] control,
    output wire [5:0] elem_data,
    output wire is_new_block
);
    assign control      = encoded_input[7:6];
    assign elem_data    = encoded_input[5:0];
    assign is_new_block = (control == 2'b01);
endmodule
