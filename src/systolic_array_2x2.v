module systolic_array_2x2 #(
    parameter WIDTH = 8
)(
    input wire clk,
    input wire rst,
    input wire clear,
    input wire activation,
    
    // Direct memory connections (data comes straight from memory!)
    input wire [7:0] weight0, weight1, weight2, weight3,
    input wire [7:0] input0, input1, input2, input3,
    
    // Control signals from merged control unit (lightweight!)
    input wire data_valid,
    input wire [1:0] a0_sel, a1_sel, b0_sel, b1_sel,
    input wire transpose,
    
    // Outputs
    output wire signed [15:0] c00, c01, c10, c11
);

    // Internal signals between PEs
    wire [WIDTH-1:0] a_wire [0:1][0:2];
    wire [WIDTH-1:0] b_wire [0:2][0:1];
    wire [WIDTH*2-1:0] c_array [0:1][0:1];

    assign a_wire[0][0] = data_valid ? 
                     (a0_sel == 2'd0) ? weight0 :
                     (a0_sel == 2'd1) ? weight1 : 8'b0 : 8'b0;

    assign a_wire[1][0] = data_valid ?
                     (a1_sel == 2'd0) ? weight2 :
                     (a1_sel == 2'd1) ? weight3 : 8'b0 : 8'b0;

    assign b_wire[0][0] = data_valid ?
                     (b0_sel == 2'd0) ? input0 :
                     (b0_sel == 2'd1) ? (transpose ? input1 : input2) : 8'b0 : 8'b0;

    assign b_wire[0][1] = data_valid ?
                     (b1_sel == 2'd0) ? (transpose ? input2 : input1) :
                     (b1_sel == 2'd1) ? input3 : 8'b0 : 8'b0;

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
