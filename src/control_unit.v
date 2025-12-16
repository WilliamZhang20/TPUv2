`default_nettype none

module control_unit (
    input wire clk,
    input wire rst,
    input wire enable,
    input wire stat_weights,
    input wire load_weights,

    // Memory address control
    output reg [2:0] mem_addr,

    // Systolic array control signals (lightweight!)
    output wire clear,
    output reg [1:0] a0_sel, a1_sel, b0_sel, b1_sel
);

    reg [2:0] mmu_cycle; // Counting Systolic Array Stages

    // Done signal and clear signal
    assign clear = (mmu_cycle == 3'b000);

    // State machine and control signal generation
    always @(posedge clk) begin
        if (rst) begin
            mmu_cycle <= 0;
            mem_addr <= 0;
        end else begin
            // Handle memory addressing
            if (enable) begin
                mem_addr <= mem_addr + 1;
            end else begin
                mem_addr <= 0;
                mmu_cycle <= 0;
            end

            if (mem_addr == 3'b101) begin
                mmu_cycle <= 0; // systolic cycling begins at 5th load
            end else begin
                mmu_cycle <= mmu_cycle + 1;
                if (mem_addr == 3'b111) begin 
                    if (!stat_weights || (stat_weights && load_weights)) begin
                        mem_addr <= 0;
                    end else if (stat_weights && !load_weights) begin
                        mem_addr <= 4;
                    end
                end
            end
        end
    end

    // Combinational logic for data_out
    always @(*) begin
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
