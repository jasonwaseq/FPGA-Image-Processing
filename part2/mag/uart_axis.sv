

module uart_axis
   #(parameter example_p = 0
    , parameter int linewidth_px_p = 480
    )
   (input [0:0] clk_i
   ,input [0:0] reset_i
   ,input [0:0] rx_serial_i
   ,output [0:0] tx_serial_o
   ,output [5:1] led_o
    );

   localparam data_width_lp = 8;
   
   wire [data_width_lp-1:0] m_axis_uart_tdata;
   wire       m_axis_uart_tvalid;
   wire       m_axis_uart_tready;

   wire [data_width_lp-1:0] s_axis_uart_tdata;
   wire       s_axis_uart_tvalid;
   wire       s_axis_uart_tready;

   wire [23:0] rgb_tdata;
   wire        rgb_tvalid;
   wire        rgb_tready;

   // UART status signals (connected but unused)
   wire tx_busy;
   wire rx_busy;
   wire rx_overrun_error;
   wire rx_frame_error;
   
   wire [31:0] s_axis_tdata;
   wire        s_axis_tvalid;
   wire        s_axis_tlast;
   wire [3:0]  s_axis_tkeep;
   wire        s_axis_tready;

   uart #(
      .DATA_WIDTH(data_width_lp)
   ) uart_inst (
      .clk(clk_i),
      .rst(reset_i),

      .s_axis_tready(s_axis_uart_tready),
      .s_axis_tvalid(s_axis_uart_tvalid),
      .s_axis_tdata(s_axis_uart_tdata),

      .m_axis_tready(m_axis_uart_tready),
      .m_axis_tvalid(m_axis_uart_tvalid),
      .m_axis_tdata(m_axis_uart_tdata),

      .rxd(rx_serial_i),
      .txd(tx_serial_o),
      .tx_busy(tx_busy),
      .rx_busy(rx_busy),
      .rx_overrun_error(rx_overrun_error),
      .rx_frame_error(rx_frame_error),
      .prescale(16'd27)
   );

   // Packs 3 bytes (R, G, B) into 24-bit word 
   reg [7:0] rgb_r, rgb_g, rgb_b;
   reg [1:0] byte_count;
   
   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         byte_count <= 2'b00;
         rgb_r <= 8'h00;
         rgb_g <= 8'h00;
         rgb_b <= 8'h00;
      end else begin
         if (m_axis_uart_tvalid && rgb_tready) begin
            case (byte_count)
               2'b00: begin
                  rgb_r <= m_axis_uart_tdata;
                  byte_count <= 2'b01;
               end
               2'b01: begin
                  rgb_g <= m_axis_uart_tdata;
                  byte_count <= 2'b10;
               end
               2'b10: begin
                  rgb_b <= m_axis_uart_tdata;
                  byte_count <= 2'b00;
               end
               default: byte_count <= 2'b00;
            endcase
         end
      end
   end
   
   // Output valid when 3 bytes collected
   assign rgb_tvalid = m_axis_uart_tvalid && (byte_count == 2'b10);
   assign m_axis_uart_tready = rgb_tready || (byte_count != 2'b10);
   assign rgb_tdata = {rgb_b, rgb_g, rgb_r};  
   
   wire [7:0] gray;
   wire gray_sobel_valid, gray_sobel_ready;
   wire sobel_mag_valid, sobel_mag_ready;
   
   rgb2gray rgb2gray_inst(
       .clk_i(clk_i)
      ,.reset_i(reset_i)
      ,.valid_i(rgb_tvalid)
      ,.ready_o(rgb_tready)
      ,.red_i(rgb_tdata[7:0])
      ,.green_i(rgb_tdata[15:8])
      ,.blue_i(rgb_tdata[23:16])
      ,.valid_o(gray_sobel_valid)
      ,.ready_i(gray_sobel_ready)
      ,.gray_o(gray)
    );

   // Gradient signals from sobel 
   wire signed [11:0] gx, gy;
   wire [15:0] sobel_data;
   sobel #(
      .width_in_p(8),
      .width_out_p(16),
      .linewidth_px_p(480)
   ) sobel_inst (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .valid_i(gray_sobel_valid),
      .ready_o(gray_sobel_ready),
      .data_i(gray),
      .gx_o(gx),
      .gy_o(gy),
      .valid_o(sobel_mag_valid),
      .ready_i(sobel_mag_ready),
      .data_o(sobel_data)
   );

   wire [15:0] mag;
   mag mag_inst(
      .clk_i(clk_i),
      .reset_i(reset_i),
      .valid_i(sobel_mag_valid),
      .gx_i({{4{gx[11]}}, gx}),  // sign-extend 12-bit gx to 16-bits
      .gy_i({{4{gy[11]}}, gy}),  // sign-extend 12-bit gy to 16-bits
      .ready_o(sobel_mag_ready),
      .valid_o(s_axis_tvalid),
      .mag_o(mag),
      .ready_i(s_axis_tready)
  );

   wire [19:0] mag_scaled = mag << 4;  // multiply by 16 for brightness
  
   // Direct 8-bit output to UART
   wire [7:0] mag_output = mag_scaled[13:6];
  
   assign s_axis_uart_tdata = mag_output;
   assign s_axis_uart_tvalid = s_axis_tvalid;
   assign s_axis_tready = s_axis_uart_tready;

   assign led_o = 5'b10101;
   
endmodule

