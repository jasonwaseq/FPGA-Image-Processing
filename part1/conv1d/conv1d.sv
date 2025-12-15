module conv1d
  #(
   // This is here to help, but we won't change it.
   parameter width_p = 24
   )
  (input [0:0] clk_i
  ,input [0:0] reset_i

  ,input [0:0] valid_i
  ,input [width_p - 1:0] data_i
  ,output [0:0] ready_o

  ,output [0:0] valid_o
  ,output [width_p - 1:0] data_o
  ,input [0:0] ready_i
  );



endmodule
