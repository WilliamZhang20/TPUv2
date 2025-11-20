`default_nettype none

module control_unit (
    input wire clk,
    input wire rst,
    input wire load_en,

    // Memory interface  
    output reg [2:0] mem_addr,

    // MMU feeding control
    output reg mmu_en,
    output reg [2:0] mmu_cycle
);

    // STATES
    localparam [1:0] S_IDLE                  = 2'b00;
    localparam [1:0] S_LOAD_MATS             = 2'b01;
    localparam [1:0] S_MMU_FEED_COMPUTE_WB   = 2'b10;

    reg [1:0] state, next_state;

    // Next state logic
    always @(*) begin
        next_state = state;

        case (state)
            S_IDLE: begin
                if (load_en) begin
                    next_state = S_LOAD_MATS;
                end
            end
            
            S_LOAD_MATS: begin
                // All 8 elements loaded (4 for each matrix)
                if (mem_addr == 3'b111) begin 
                    next_state = S_MMU_FEED_COMPUTE_WB;
                end
            end
            
            S_MMU_FEED_COMPUTE_WB:
                next_state = S_MMU_FEED_COMPUTE_WB;
               /* MMU CYCLE PATTERN
                * Cycle 0: Start feeding data (a00×b00 starts)
                * Cycle 1: First partial products computed, more data fed
                * Cycle 2: c00 ready (a00×b00 + a01×b10), is output, while take in next a00
                * Cycle 3: c01 and c10 ready simultaneously, read next a01:
                *          c01 = a00×b01 + a01×b11
                *          c10 = a10×b00 + a11×b10
                * Cycle 4: c11 ready (a10×b01 + a11×b11), read next a10
                * Cycle 5: All outputs remain valid, read next a11
                * Cycle 6: Keep outputting, read next b00
                * Cycle 7: Keep outputting, read next b01
                * Back to cycle 0: Start feeding data (a00×b00 starts), keep outputting, read next b10
                * Cycle 1: First partial products computed, keep outputting, read next b01
                * Cycle 2: c00 ready, begin output, take in next a00, pattern continues...
                */

			default: begin
				next_state = S_IDLE;
			end
        endcase
    end

    // State Machine
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            mmu_cycle <= 0;
            mmu_en <= 0;
            mem_addr <= 0;
        end else begin
            state <= next_state;
            mem_addr <= 0;
            case (state)
                S_IDLE: begin
                    mmu_cycle <= 0;
                    mmu_en <= 0;
                    if (load_en) begin
                        mem_addr <= mem_addr + 1;
                    end
                end

                S_LOAD_MATS: begin
                    if (load_en) begin
                        mem_addr <= mem_addr + 1;
                    end

                    if (mem_addr == 3'b101) begin
                        mmu_en <= 1;
                    end else if (mem_addr >= 3'b110) begin
                        mmu_en <= 1;
                        mmu_cycle <= mmu_cycle + 1;
                        if (mem_addr == 3'b111) begin 
                            mem_addr <= 0;
                        end
                    end
                end

                S_MMU_FEED_COMPUTE_WB: begin
                    // Now: the TPU will be forever stuck in this cycle...
                    // Cycles through counter of 8...
                    // In each cycle of 8 counts, it will: output 4 16-bit output elements the result of the previous matmul,
                    // and take in 8 new 8-bit elements
                    if (load_en) begin
                        mem_addr <= mem_addr + 1;
                    end
					mmu_cycle <= mmu_cycle + 1; // allow mmu_cycle to continue incrementing, permitting a pipeline flush
                    if (mmu_cycle == 3'b111) begin
                        mmu_cycle <= 0;
                    end else if (mmu_cycle == 1) begin
                        mem_addr <= 0;
                    end
                end
				
				default: begin
                    mmu_cycle <= 0;
                    mmu_en <= 0;
				end
            endcase
        end
    end

endmodule