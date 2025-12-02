`default_nettype none
`timescale 1ns / 1ps

/* This testbench just instantiates the module and makes some convenient wires
   that can be driven / tested by the cocotb test.py.
*/
module tb ();

  // Dump the signals to a VCD file. You can view it with gtkwave or surfer.
  initial begin
`ifdef VCD_PATH
    $dumpfile(`VCD_PATH);
`else
		$dumpfile("tb.vcd");
`endif
		$dumpvars(0, tb);
        #1;
	end

  // Wire up the inputs and outputs:
  reg clk;
  reg rst;
  reg clear;
  reg [7:0] a_in;
  reg [7:0] b_in;
  wire [7:0] a_out;
  wire [7:0] b_out;
  wire [15:0] c_out;

  // Replace tt_um_example with your module name:
  PE dut (
      .clk    (clk),      // clock
      .rst    (rst),    // reset
      .clear  (clear),
      .a_in   (a_in),
      .b_in   (b_in),
      .a_out  (a_out),
      .b_out  (b_out),
      .c_out  (c_out)
  );

endmodule