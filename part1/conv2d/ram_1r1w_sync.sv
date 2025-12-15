`ifndef BINPATH
 `define BINPATH ""
`endif
module ram_1r1w_sync
  #(parameter [31:0] width_p = 8
  ,parameter [31:0] depth_p = 512
  /* verilator lint_off UNUSEDPARAM */
  ,parameter string filename_p = "memory_init_file.bin")
  /* verilator lint_on UNUSEDPARAM */
  (input [0:0] clk_i
  ,input [0:0] reset_i

  ,input [0:0] wr_valid_i
  ,input [width_p-1:0] wr_data_i
  ,input [$clog2(depth_p) - 1 : 0] wr_addr_i

  ,input [0:0] rd_valid_i
  ,input [$clog2(depth_p) - 1 : 0] rd_addr_i
  ,output [width_p-1:0] rd_data_o);

  logic [width_p-1:0] mem [depth_p-1:0];
  
  logic [width_p-1:0] rd_data_l;

  always_ff @(posedge clk_i) begin
    if (~reset_i && rd_valid_i)
      rd_data_l <= mem[rd_addr_i];
    if (~reset_i && wr_valid_i) 
      mem[wr_addr_i] <= wr_data_i;
  end

  assign rd_data_o = rd_data_l;

   initial begin
      // Display depth and width (You will need to match these in your init file)
      $display("%m: depth_p is %d, width_p is %d", depth_p, width_p);
      // Initialize memory to zero for Icarus simulator compatibility
      for (int i = 0; i < depth_p; i++) begin
        mem[i] = '0;
        $dumpvars(0, mem[i]);
      end
   end

endmodule
