module conv2d
  #(
   // This is the number of pixels in each line of the image
    parameter  linewidth_px_p = 480
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
      // expose the 3x3 kernel window to the consumer (top-left is kernel[0][0])
      ,output logic [width_p-1:0] kernel [2:0][2:0]
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
   
   logic [addr_width_lp-1:0] wr_ptr1, rd_ptr1;
   logic [addr_width_lp-1:0] wr_ptr2, rd_ptr2;
   
   // 9 registers: 3 per line
   logic [width_p-1:0] mem [2:0][2:0];
   
   logic [width_p-1:0] ram_rd_data1, ram_rd_data2;
   
   // delay buffer 1
   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         wr_ptr1 <= '0;
      end else if (enable_w) begin
         if (wr_ptr1 == linewidth_px_p[addr_width_lp-1:0] - 1)
            wr_ptr1 <= '0;
         else
            wr_ptr1 <= wr_ptr1 + 1;
      end
   end
   
   // delay buffer 2
   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         wr_ptr2 <= '0;
      end else if (enable_w) begin
         if (wr_ptr2 == linewidth_px_p[addr_width_lp-1:0] - 1)
            wr_ptr2 <= '0;
         else
            wr_ptr2 <= wr_ptr2 + 1;
      end
   end
   
   // Read pointers delayed by delay_per_section_lp
   localparam [addr_width_lp-1:0] offset_lp = delay_per_section_lp[addr_width_lp-1:0];
   
   always_comb begin
      if (wr_ptr1 >= offset_lp)
         rd_ptr1 = wr_ptr1 - offset_lp;
      else
         rd_ptr1 = wr_ptr1 + (linewidth_px_p[addr_width_lp-1:0] - offset_lp);
      if (wr_ptr2 >= offset_lp)
         rd_ptr2 = wr_ptr2 - offset_lp;
      else
         rd_ptr2 = wr_ptr2 + (linewidth_px_p[addr_width_lp-1:0] - offset_lp);
   end
   
   logic [addr_width_lp-1:0] ram_rd_addr1, ram_rd_addr2;
   
   always_comb begin
      if (enable_w) begin
         ram_rd_addr1 = (rd_ptr1 == linewidth_px_p[addr_width_lp-1:0] - 1) ? '0 : rd_ptr1 + 1;
         ram_rd_addr2 = (rd_ptr2 == linewidth_px_p[addr_width_lp-1:0] - 1) ? '0 : rd_ptr2 + 1;
      end else begin
         ram_rd_addr1 = rd_ptr1;
         ram_rd_addr2 = rd_ptr2;
      end
   end
   
   // delay from mem[0][0] to mem[1][2]
   ram_1r1w_sync #(
      .width_p(width_p),
      .depth_p(linewidth_px_p),
      .filename_p("")
   ) delay_buffer_1 (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .wr_valid_i(enable_w),
      .wr_data_i(mem[0][0]),
      .wr_addr_i(wr_ptr1),
      .rd_valid_i(1'b1),
      .rd_addr_i(ram_rd_addr1),
      .rd_data_o(ram_rd_data1)
   );
   
   // delay from mem[1][0] to mem[2][2]
   ram_1r1w_sync #(
      .width_p(width_p),
      .depth_p(linewidth_px_p),
      .filename_p("")
   ) delay_buffer_2 (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .wr_valid_i(enable_w),
      .wr_data_i(mem[1][0]),
      .wr_addr_i(wr_ptr2),
      .rd_valid_i(1'b1),
      .rd_addr_i(ram_rd_addr2),
      .rd_data_o(ram_rd_data2)
   );
   
   // Shift register
   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         mem[0][0] <= '0;
         mem[0][1] <= '0;
         mem[0][2] <= '0;
         mem[1][0] <= '0;
         mem[1][1] <= '0;
         mem[1][2] <= '0;
         mem[2][0] <= '0;
         mem[2][1] <= '0;
         mem[2][2] <= '0;
      end else if (enable_w) begin
         // Line 0: newest line
         mem[0][0] <= mem[0][1];
         mem[0][1] <= mem[0][2];
         mem[0][2] <= data_i;
         // Line 1 from RAM delay buffer 1
         mem[1][0] <= mem[1][1];
         mem[1][1] <= mem[1][2];
         mem[1][2] <= ram_rd_data1;
         // Line 2 from RAM delay buffer 2
         mem[2][0] <= mem[2][1];
         mem[2][1] <= mem[2][2];
         mem[2][2] <= ram_rd_data2;
      end
   end
   
   // kernel mapping 
   assign kernel[0][2] = mem[2][2];
   assign kernel[0][1] = mem[2][1];
   assign kernel[0][0] = mem[2][0];

   assign kernel[1][2] = mem[1][2];
   assign kernel[1][1] = mem[1][1];
   assign kernel[1][0] = mem[1][0];

   assign kernel[2][2] = mem[0][2];
   assign kernel[2][1] = mem[0][1];
   assign kernel[2][0] = mem[0][0];

   localparam int ext_width = (2 * width_p) - width_p; 
   assign data_o = ({{ext_width{1'b0}}, kernel[2][2]} + {{ext_width{1'b0}}, kernel[2][1]} + {{ext_width{1'b0}}, kernel[2][0]}) +
                   ({{ext_width{1'b0}}, kernel[1][2]} + {{ext_width{1'b0}}, kernel[1][1]} + {{ext_width{1'b0}}, kernel[1][0]}) +
                   ({{ext_width{1'b0}}, kernel[0][2]} + {{ext_width{1'b0}}, kernel[0][1]} + {{ext_width{1'b0}}, kernel[0][0]});
                   
endmodule
