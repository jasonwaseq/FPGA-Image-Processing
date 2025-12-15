// This is yosys' simulation model of the LUT6
(* abc9_lut=5 *)
module LUT6(output O, input I0, I1, I2, I3, I4, I5);
  parameter [63:0] INIT = 0;
  wire [31: 0] s5 = I5 ? INIT[63:32] : INIT[31: 0];
  wire [15: 0] s4 = I4 ?   s5[31:16] :   s5[15: 0];
  wire [ 7: 0] s3 = I3 ?   s4[15: 8] :   s4[ 7: 0];
  wire [ 3: 0] s2 = I2 ?   s3[ 7: 4] :   s3[ 3: 0];
  wire [ 1: 0] s1 = I1 ?   s2[ 3: 2] :   s2[ 1: 0];
  assign O = I0 ? s1[1] : s1[0];
  specify
    (I0 => O) = 642;
    (I1 => O) = 631;
    (I2 => O) = 472;
    (I3 => O) = 407;
    (I4 => O) = 238;
    (I5 => O) = 127;
  endspecify
endmodule
