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

    wire mmu_en; // internal signal
    reg clear; // reset of PEs only
    wire [2:0] mem_addr; // 3-bit address for matrix and element selection
    
    wire [2:0] mmu_cycle; // compute/output cycle count - 3 bit

    wire [7:0] weight0, weight1, weight2, weight3;
    wire [7:0] input0, input1, input2, input3;

    wire [15:0] outputs [0:3]; // raw accumulations (16-bit)
    wire [7:0] out_data; // sent to CPU
    // Ports of the systolic Array
    wire [7:0] a_data0, b_data0, a_data1, b_data1;

    wire done;
    wire [1:0] state;

    wire [7:0] stage1, stage2, stage3;

    wire [7:0] uio_s1, uio_s2, uio_s3, uio_s4;

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

    control_unit central_ctrl (
        .clk(clk),
        .rst(~rst_n),
        .load_en(load_en),
        .mem_addr(mem_addr),
        .mmu_en(mmu_en),
        .mmu_cycle(mmu_cycle),
        .state_out(state)
    );

    systolic_array_2x2 mmu (
        .clk(clk),
        .rst(~rst_n),
        .clear(clear),
        .activation(activation),
        .a_data0(a_data0),
        .a_data1(a_data1),
        .b_data0(b_data0),
        .b_data1(b_data1),
        .c00(outputs[0]), 
        .c01(outputs[1]), 
        .c10(outputs[2]), 
        .c11(outputs[3])
    );

    mmu_feeder compute_ctrl (
        .clk(clk),
        .rst(~rst_n),
        .en(mmu_en),
        .mmu_cycle(mmu_cycle),
        .transpose(transpose),
        .weight0(weight0), .weight1(weight1), .weight2(weight2), .weight3(weight3),
        .input0(input0), .input1(input1), .input2(input2), .input3(input3),
        .c00(outputs[0]), 
        .c01(outputs[1]), 
        .c10(outputs[2]), 
        .c11(outputs[3]),
        .clear(clear),
        .a_data0(a_data0),
        .a_data1(a_data1),
        .b_data0(b_data0),
        .b_data1(b_data1),
        .done(done),
        .host_outdata(out_data)
    );

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : buf_loop
            (* keep *) buffer buf1 (.A(out_data[i]), .X(stage1[i]));
            (* keep *) buffer buf2 (.A(stage1[i]), .X(stage2[i]));
            (* keep *) buffer buf3 (.A(stage2[i]), .X(stage3[i]));
        end
    endgenerate
    assign uo_out = stage3;

    assign uio_s1 = {done, state, 5'b0};

    genvar j;
    generate
        for (j = 0; j < 8; j = j + 1) begin : uio_bufs
            (* keep *) buffer buf21 (.A(uio_s1[j]), .X(uio_s2[j]));
            (* keep *) buffer buf22 (.A(uio_s2[j]), .X(uio_s3[j]));
            (* keep *) buffer buf23 (.A(uio_s3[j]), .X(uio_s4[j]));
        end
    endgenerate
    assign uio_out = uio_s4;
    assign uio_oe = 8'b11100000;

    wire _unused = &{ena, uio_in[7:3]};

endmodule
