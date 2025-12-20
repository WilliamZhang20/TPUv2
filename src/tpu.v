/*
 * Copyright (c) 2025 William
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_tpu (
    input  wire [7:0] ui_in,      // data input
    output reg [7:0] uo_out,     // data output
    input  wire [7:0] uio_in,     // control input
    output reg [7:0] uio_out,   // bidirectional pin output control
    output reg [7:0] uio_oe, 
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    wire load_en = uio_in[0];
    wire transpose = uio_in[1];
    wire activation = uio_in[2];
    wire elemwise = uio_in[3];
    wire enable = uio_in[4];
    wire stat_weights = uio_in[5];
    wire load_weights = uio_in[6];

    wire [2:0] mem_addr; // 3-bit address for matrix and element selection

    wire [7:0] weight0, weight1, weight2, weight3;
    wire [7:0] input0, input1, input2, input3;

    wire [15:0] outputs [0:3]; // raw accumulations (16-bit)

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
        .enable(enable),
        .stat_weights(stat_weights),
        .load_weights(load_weights),
        .mem_addr(mem_addr),
        .clear(clear),
        .a0_sel(a0_sel), .a1_sel(a1_sel), .b0_sel(b0_sel), .b1_sel(b1_sel)
    );

    systolic_array_2x2 mmu (
        .clk(clk),
        .rst(~rst_n),
        .clear(clear),
        .activation(activation),
        .elemwise(elemwise),
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

    // Buffer of output after clearing previous
    reg [15:0] tail_hold;
    reg [15:0] hold2;

    always @(posedge clk) begin
        if (mem_addr == 3'b101) begin
            hold2 <= outputs[2];
            tail_hold <= {8'b0, outputs[3][7:0]};
        end else if (mem_addr == 3'b110) begin
            if (stat_weights) begin
                tail_hold <= outputs[3];
            end else begin
                hold2 <= 0;
                tail_hold <= {8'b0, outputs[3][7:0]};
            end
        end
    end

    always @(*) begin
        uo_out  = 8'b0;
        uio_out = 8'b0;
        uio_oe  = 8'b0;

        if (!stat_weights) begin
            uio_oe = 8'b00000000;
            case (mem_addr)
                3'b000: uo_out = outputs[0][15:8];
                3'b001: uo_out = outputs[0][7:0];
                3'b010: uo_out = outputs[1][15:8];
                3'b011: uo_out = outputs[1][7:0];
                3'b100: uo_out = outputs[2][15:8];
                3'b101: uo_out = outputs[2][7:0];
                3'b110: uo_out = outputs[3][15:8];
                3'b111: uo_out = tail_hold[7:0];
            endcase
        end else begin
            uio_oe = 8'b11111111;
            case (mem_addr)
                3'b100: begin
                    uo_out  = outputs[0][15:8];
                    uio_out = outputs[0][7:0];
                end
                3'b101: begin
                    uo_out  = outputs[1][15:8];
                    uio_out = outputs[1][7:0];
                end
                3'b110: begin
                    uo_out  = hold2[15:8];
                    uio_out = hold2[7:0];
                end
                3'b111: begin
                    uo_out  = tail_hold[15:8];
                    uio_out = tail_hold[7:0];
                end
            endcase
        end
    end

    wire _unused = &{ena, uio_in[7:3]};

endmodule
