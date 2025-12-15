// This module implements a PMOS transistor
module pmosfet
  (input [0:0] gate_i
  ,input [0:0] source_i
  ,output [0:0] drain_o);

   assign drain_o = gate_i ? 1'bz : source_i;

endmodule
	   
