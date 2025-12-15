module conv2d
  #(
   // This is the number of pixels in each line of the image
    parameter  linewidth_px_p = 16
    // This is the width of each input pixel. You will use 8 for part 2.
    // We will only change it to help with debugging (so that 0 to 4 * 480 can be represented without overflow)
    ,parameter width_p = 8)
   (input [0:0] clk_i
    ,input [0:0] reset_i
    ,input [0:0] valid_i
    ,output [0:0] ready_o
    ,input [width_p - 1 : 0] data_i
    ,output [0:0] valid_o
    ,input [0:0] ready_i
    // Just to make our lives easy, we're going to double the output width.
    ,output [(2 * width_p) - 1 : 0] data_o
    );

   // Elastic State Machine Logic
   logic valid_r;
   wire enable_w;
   assign enable_w = valid_i & ready_o;

   assign ready_o = ~valid_o | ready_i;

   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         valid_r <= 1'b0;
      end else if (ready_o) begin
         valid_r <= enable_w;
      end
   end
   assign valid_o = valid_r;

   localparam delay_per_section_lp = linewidth_px_p - 3;
   localparam addr_width_lp = $clog2(linewidth_px_p);
   
   logic [addr_width_lp-1:0] wr_ptr, rd_ptr;
   
   // 9 registers: 3 per line
   logic [width_p-1:0] kernel [2:0][2:0];
   
   logic [2*width_p-1:0] ram_rd_data;
   
   // Single write pointer for both buffers
   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         wr_ptr <= '0;
      end else if (enable_w) begin
         if (wr_ptr == linewidth_px_p[addr_width_lp-1:0] - 1)
            wr_ptr <= '0;
         else
            wr_ptr <= wr_ptr + 1;
      end
   end
   
   always_comb begin
      if (wr_ptr >= delay_per_section_lp[addr_width_lp-1:0])
         rd_ptr = wr_ptr - delay_per_section_lp[addr_width_lp-1:0];
      else
         rd_ptr = wr_ptr + (linewidth_px_p[addr_width_lp-1:0] - delay_per_section_lp[addr_width_lp-1:0]);
   end
   
   logic [addr_width_lp-1:0] ram_rd_addr;
   
   always_comb begin
      if (enable_w) begin
         ram_rd_addr = (rd_ptr == linewidth_px_p[addr_width_lp-1:0] - 1) ? '0 : rd_ptr + 1;
      end else begin
         ram_rd_addr = rd_ptr;
      end
   end
   
   ram_1r1w_sync #(
      .width_p(2*width_p),
      .depth_p(linewidth_px_p),
      .filename_p("")
   ) delay_buffer (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .wr_valid_i(enable_w),
      .wr_data_i({kernel[1][0], kernel[0][0]}), 
      .wr_addr_i(wr_ptr),
      .rd_valid_i(1'b1),
      .rd_addr_i(ram_rd_addr),
      .rd_data_o(ram_rd_data)
   );
   
   // Shift register
   always_ff @(posedge clk_i) begin
      if (enable_w) begin
         // Line 0: newest line
         kernel[0][0] <= kernel[0][1];
         kernel[0][1] <= kernel[0][2];
         kernel[0][2] <= data_i;
         // Line 1 from RAM delay buffer (lower bits)
         kernel[1][0] <= kernel[1][1];
         kernel[1][1] <= kernel[1][2];
         kernel[1][2] <= ram_rd_data[width_p-1:0];
         // Line 2 from RAM delay buffer (upper bits)
         kernel[2][0] <= kernel[2][1];
         kernel[2][1] <= kernel[2][2];
         kernel[2][2] <= ram_rd_data[2*width_p-1:width_p];
      end
   end
   
   assign data_o = ((2*width_p)'(kernel[0][0]) + (2*width_p)'(kernel[0][1]) + (2*width_p)'(kernel[0][2])) +
                   ((2*width_p)'(kernel[1][0]) + (2*width_p)'(kernel[1][1]) + (2*width_p)'(kernel[1][2])) +
                   ((2*width_p)'(kernel[2][0]) + (2*width_p)'(kernel[2][1]) + (2*width_p)'(kernel[2][2]));
                   
endmodule
