module PE (
    input  wire        clk,
    input  wire        rst,
    input  wire        clear,
    input  wire [7:0]  a_in,
    input  wire [7:0]  b_in,
    output reg  [7:0]  a_out,
    output reg  [7:0]  b_out,
    output reg  [15:0] c_out
);
    localparam int ELEM_WIDTH = 6;
    localparam int MANT_WIDTH = 5;
    
    // ========================================================================
    // Input Decoding
    // ========================================================================
    // Extract control bits and element data
    wire [1:0] a_ctrl = a_in[7:6];
    wire [1:0] b_ctrl = b_in[7:6];
    wire [ELEM_WIDTH-1:0] a_elem = a_in[ELEM_WIDTH-1:0];
    wire [ELEM_WIDTH-1:0] b_elem = b_in[ELEM_WIDTH-1:0];
    
    // Control bits interpretation:
    // [7:6] = 00: normal operation, use previous exponent
    // [7:6] = 01: new block, exponent in elem[5:0] (special encoding)
    // [7:6] = 10: exponent update, [5:0] = exponent delta
    // [7:6] = 11: flush/control signal
    
    // ========================================================================
    // Exponent Management
    // ========================================================================
    reg [7:0] current_exp_a;
    reg [7:0] current_exp_b;
    wire new_block_a = (a_ctrl == 2'b01);
    wire new_block_b = (b_ctrl == 2'b01);
    wire new_block = new_block_a | new_block_b;
    
    // Extract exponents (in real design, would come from block header)
    // For now, use fixed BF16 exponent range
    wire [7:0] exp_a = current_exp_a;
    wire [7:0] exp_b = current_exp_b;
    
    // ========================================================================
    // STAGE 1: Decode
    // ========================================================================
    wire a_sign, b_sign;
    wire [MANT_WIDTH-1:0] a_mant, b_mant;
    
    mx_decoder #(.ELEM_WIDTH(ELEM_WIDTH)) dec_a (
        .mx_elem(a_elem),
        .sign(a_sign),
        .mantissa(a_mant)
    );
    
    mx_decoder #(.ELEM_WIDTH(ELEM_WIDTH)) dec_b (
        .mx_elem(b_elem),
        .sign(b_sign),
        .mantissa(b_mant)
    );
    
    // Stage 1 → 2 pipeline
    reg a_sign_s1, b_sign_s1;
    reg [MANT_WIDTH-1:0] a_mant_s1, b_mant_s1;
    reg [7:0] exp_a_s1, exp_b_s1;
    reg new_block_s1;
    
    // ========================================================================
    // STAGE 2: Multiply
    // ========================================================================
    wire prod_sign;
    wire [8:0] prod_exp;
    wire [2*MANT_WIDTH+1:0] prod_mant;
    
    mx_multiplier #(.MANT_WIDTH(MANT_WIDTH)) mul (
        .sign_a(a_sign_s1),
        .sign_b(b_sign_s1),
        .mant_a(a_mant_s1),
        .mant_b(b_mant_s1),
        .exp_a(exp_a_s1),
        .exp_b(exp_b_s1),
        .prod_sign(prod_sign),
        .prod_exp(prod_exp),
        .prod_mant(prod_mant)
    );
    
    // Stage 2 → 3 pipeline
    reg prod_sign_s2;
    reg [8:0] prod_exp_s2;
    reg [2*MANT_WIDTH+1:0] prod_mant_s2;
    reg new_block_s2;
    
    // ========================================================================
    // STAGE 3: Accumulate
    // ========================================================================
    wire [15:0] acc_next;
    
    mx_accumulator #(.PROD_WIDTH(2*MANT_WIDTH+2)) acc (
        .prod_sign(prod_sign_s2),
        .prod_exp(prod_exp_s2),
        .prod_mant(prod_mant_s2),
        .acc_in(c_out),
        .new_block(new_block_s2),
        .acc_out(acc_next)
    );
    
    // ========================================================================
    // Pipeline Control
    // ========================================================================
    always @(posedge clk) begin
        // Systolic data flow - pass through inputs
        a_out <= a_in;
        b_out <= b_in;
        
        if (rst) begin
            // Stage 1
            a_sign_s1 <= 0;
            b_sign_s1 <= 0;
            a_mant_s1 <= 0;
            b_mant_s1 <= 0;
            exp_a_s1 <= 0;
            exp_b_s1 <= 0;
            new_block_s1 <= 0;
            
            // Stage 2
            prod_sign_s2 <= 0;
            prod_exp_s2 <= 0;
            prod_mant_s2 <= 0;
            new_block_s2 <= 0;
            
            // Accumulator
            c_out <= 16'd0;
            
            // Exponent tracking
            current_exp_a <= 8'd127;  // BF16 bias
            current_exp_b <= 8'd127;
        end
        else begin
            // Update current exponents on new block
            if (new_block_a) begin
                // In real design, would decode exponent from stream
                current_exp_a <= 8'd127;  // Placeholder
            end
            if (new_block_b) begin
                current_exp_b <= 8'd127;  // Placeholder
            end
            
            // Stage 1 → 2
            a_sign_s1 <= a_sign;
            b_sign_s1 <= b_sign;
            a_mant_s1 <= a_mant;
            b_mant_s1 <= b_mant;
            exp_a_s1 <= exp_a;
            exp_b_s1 <= exp_b;
            new_block_s1 <= new_block;
            
            // Stage 2 → 3
            prod_sign_s2 <= prod_sign;
            prod_exp_s2 <= prod_exp;
            prod_mant_s2 <= prod_mant;
            new_block_s2 <= new_block_s1;
            
            // Accumulator
            if (clear)
                c_out <= 16'd0;
            else
                c_out <= acc_next;
        end
    end

endmodule
