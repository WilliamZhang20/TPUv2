module buffer (
    output wire X   ,
    input  wire A   ,
    input  wire VPWR,
    input  wire VGND,
    input  wire VPB ,
    input  wire VNB
);

    assign X = A;

    wire _unused;
    assign _unused = &{ 1'b0, VPWR, VGND, VPB, VNB };

endmodule