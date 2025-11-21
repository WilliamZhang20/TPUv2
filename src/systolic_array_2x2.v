module systolic_array_2x2 #(
    parameter WIDTH = 8
)(
    input wire clk,
    input wire rst,
    input wire clear,
    input wire activation,

    input wire [WIDTH-1:0] a_data0,
    input wire [WIDTH-1:0] a_data1,
    input wire [WIDTH-1:0] b_data0,
    input wire [WIDTH-1:0] b_data1,

    output wire [WIDTH*2-1:0] c00,
    output wire [WIDTH*2-1:0] c01,
    output wire [WIDTH*2-1:0] c10,
    output wire [WIDTH*2-1:0] c11
);

    // Internal signals between PEs
    wire [WIDTH-1:0] a_wire [0:1][0:2];
    wire [WIDTH-1:0] b_wire [0:2][0:1];
    wire signed [WIDTH*2-1:0] c_array [0:1][0:1];

    // Input loading at top-left
    assign a_wire[0][0] = a_data0;
    assign a_wire[1][0] = a_data1;
    assign b_wire[0][0] = b_data0;
    assign b_wire[0][1] = b_data1;

    genvar i, j;
    generate
        for (i = 0; i < 2; i = i + 1) begin : row
            for (j = 0; j < 2; j = j + 1) begin : col
                PE pe_inst (
                    .clk(clk),
                    .rst(rst),
                    .clear(clear),
                    .a_in(a_wire[i][j]),
                    .b_in(b_wire[i][j]),
                    .a_out(a_wire[i][j+1]),
                    .b_out(b_wire[i+1][j]),
                    .c_out(c_array[i][j])
                );
            end
        end
    endgenerate

    // Combinational logic for output with optional ReLU
    assign c00 = activation ? (c_array[0][0] < 0 ? 0 : c_array[0][0]) : c_array[0][0];
    assign c01 = activation ? (c_array[0][1] < 0 ? 0 : c_array[0][1]) : c_array[0][1];
    assign c10 = activation ? (c_array[1][0] < 0 ? 0 : c_array[1][0]) : c_array[1][0];
    assign c11 = activation ? (c_array[1][1] < 0 ? 0 : c_array[1][1]) : c_array[1][1];

endmodule
