`default_nettype none

module control_unit (
    input wire clk,
    input wire rst,
    input wire load_en,

    // Systolic array feedback for output selection
    input wire signed [15:0] c00, c01, c10, c11,

    // Memory address control
    output reg [2:0] mem_addr,

    // Systolic array control signals (lightweight!)
    output wire clear,
    output reg [1:0] a0_sel, a1_sel, b0_sel, b1_sel,

    // Output interface
    output reg [7:0] data_out
);

    reg [2:0] mmu_cycle; // Counting Systolic Array Stages

    // Done signal and clear signal
    assign clear = (mmu_cycle == 3'b000);

    // Buffer of output after clearing previous
    reg [7:0] tail_hold;

    // State machine and control signal generation
    always @(posedge clk) begin
        if (rst) begin
            mmu_cycle <= 0;
            mem_addr <= 0;
            tail_hold <= 8'b0;
        end else begin
            // Handle memory addressing
            if (load_en) begin
                mem_addr <= mem_addr + 1;
            end else begin
                mem_addr <= 0;
                mmu_cycle <= 0;
            end

            if (mem_addr == 3'b101) begin
                mmu_cycle <= 0; // systolic cycling begins at 5th load
                tail_hold <= c11[7:0];
            end else begin
                mmu_cycle <= mmu_cycle + 1;
                if (mem_addr == 3'b111) begin 
                    mem_addr <= 0;
                end
            end
        end
    end

    // Combinational logic for data_out
    always @(*) begin
        data_out = 8'b0;
        case (mem_addr)
            3'b000: data_out = c00[15:8];
            3'b001: data_out = c00[7:0];
            3'b010: data_out = c01[15:8];
            3'b011: data_out = c01[7:0];
            3'b100: data_out = c10[15:8];
            3'b101: data_out = c10[7:0];
            3'b110: data_out = c11[15:8];
            3'b111: data_out = tail_hold;
        endcase

        // Generate mux selects based on mmu_cycle (same for all cycles)
        case (mmu_cycle)
            3'd0: begin
                a0_sel = 2'd0; // weight0
                a1_sel = 2'd2; // not used
                b0_sel = 2'd0; // input0
                b1_sel = 2'd2; // not used
            end
            3'd1: begin
                a0_sel = 2'd1; // weight1
                a1_sel = 2'd0; // weight2
                b0_sel = 2'd1; // input1/input2 (transpose)
                b1_sel = 2'd0; // input2/input1 (transpose)
            end
            3'd2: begin
                a0_sel = 2'd2; // not used
                a1_sel = 2'd1; // weight3
                b0_sel = 2'd2; // not used
                b1_sel = 2'd1; // input3
            end
            default: begin // by default turn everything off, i.e. set systolic inputs to 0
                a0_sel = 2'd2;
                a1_sel = 2'd2;
                b0_sel = 2'd2;
                b1_sel = 2'd2;
            end
        endcase
    end

endmodule
