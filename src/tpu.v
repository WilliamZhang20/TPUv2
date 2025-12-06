/*
 * Copyright (c) 2025 William
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_tpu (
    input  wire [7:0] ui_in,      // data input
    output wire [7:0] uo_out,     // data output (lower 8 bits of result)
    input  wire [7:0] uio_in,     // control input
    output wire [7:0] uio_out,    // done signal on uio_out[7]
    output wire [7:0] uio_oe,     // only uio_out[7] driven
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    wire load_en = uio_in[0];
    wire transpose = uio_in[1];
    wire activation = uio_in[2];
    wire [2:0] mem_addr; // 3-bit address for matrix and element selection

    wire [7:0] weight0, weight1, weight2, weight3;
    wire [7:0] input0, input1, input2, input3;

    wire [15:0] outputs [0:3]; // raw accumulations (16-bit)
    wire [7:0] out_data; // sent to CPU

    // Control signals
    wire clear;
    wire [1:0] a0_sel, a1_sel, b0_sel, b1_sel;

    // Module Instantiations
    memory mem (
        .clk(clk),
        .rst(~rst_n),
        .load_en(load_en),
        .addr(mem_addr),
        .in_data(ui_in),
        .weight0(weight0), .weight1(weight1), .weight2(weight2), .weight3(weight3),
        .input0(input0), .input1(input1), .input2(input2), .input3(input3)
    );

    control_unit control (
        .clk(clk),
        .rst(~rst_n),
        .load_en(load_en),
        .c00(outputs[0]), .c01(outputs[1]), .c10(outputs[2]), .c11(outputs[3]),
        .mem_addr(mem_addr),
        .clear(clear),
        .a0_sel(a0_sel), .a1_sel(a1_sel), .b0_sel(b0_sel), .b1_sel(b1_sel),
        .data_out(out_data)
    );

    systolic_array_2x2 mmu (
        .clk(clk),
        .rst(~rst_n),
        .clear(clear),
        .activation(activation),
        .weight0(weight0), .weight1(weight1), .weight2(weight2), .weight3(weight3),
        .input0(input0), .input1(input1), .input2(input2), .input3(input3),
        .a0_sel(a0_sel),
        .a1_sel(a1_sel),
        .b0_sel(b0_sel),
        .b1_sel(b1_sel),
        .transpose(transpose),
        .c00(outputs[0]), 
        .c01(outputs[1]), 
        .c10(outputs[2]), 
        .c11(outputs[3])
    );

    assign uo_out = out_data;
    
    assign uio_out = {8'b0};
    assign uio_oe = 8'b00000000;

    wire _unused = &{ena, uio_in[7:3]};

endmodule
