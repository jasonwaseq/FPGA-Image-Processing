`timescale 1ns / 1ps
module uart_axis
   #(parameter example_p = 0)
   (input [0:0] clk_i
  ,input [0:0] reset_i
  ,input [0:0] rx_serial_i
  ,output [0:0] tx_serial_o
  ,output [5:1] led_o
   );

   localparam [31:0] data_width_lp = 8;
   
   wire [data_width_lp-1:0] m_axis_uart_tdata;
   wire       m_axis_uart_tvalid;
   wire       m_axis_uart_tready;

   wire [data_width_lp-1:0] s_axis_uart_tdata;
   wire       s_axis_uart_tvalid;
   wire       s_axis_uart_tready;

   wire tx_busy;
   wire rx_busy;
   wire rx_overrun_error;
   wire rx_frame_error;

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

   // Packs 3 bytes (R, G, B) into a 24-bit word 
   logic [7:0] rgb_r, rgb_g, rgb_b;
   logic [1:0] byte_count;
   wire [23:0] rgb_tdata;
   wire rgb_tvalid;
   wire rgb_tready;
   
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
   assign rgb_tdata = {rgb_b, rgb_g, rgb_r};  // BGR ordering 

   // Extract RGB from 24-bit data (R, G, B)
   wire [7:0] red_i = rgb_tdata[7:0];
   wire [7:0] green_i = rgb_tdata[15:8];
   wire [7:0] blue_i = rgb_tdata[23:16];
   wire valid_i = rgb_tvalid;
   wire ready_o;
   assign rgb_tready = ready_o;

   // RGB to Grayscale
   wire [7:0] gray;
   wire gray_valid;
   wire gray_ready;

   rgb2gray rgb2gray_inst (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .valid_i(valid_i),
      .red_i(red_i),
      .green_i(green_i),
      .blue_i(blue_i),
      .ready_o(ready_o),
      .valid_o(gray_valid),
      .gray_o(gray),
      .ready_i(gray_ready)
   );

   // Sobel edge detection
   wire signed [11:0] gx, gy;
   wire [7:0] sobel_data_o;
   wire sobel_valid;
   wire sobel_ready;

   sobel #(
      .width_in_p(8),
      .width_out_p(16),
      .linewidth_px_p(480)
   ) sobel_inst (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .valid_i(gray_valid),
      .ready_o(gray_ready),
      .data_i(gray),
      .gx_o(gx),
      .gy_o(gy),
      .valid_o(sobel_valid),
      .ready_i(sobel_ready),
      .data_o(sobel_data_o)
   );

   // Magnitude computation
   wire [15:0] mag;
   wire mag_valid;
   wire mag_ready;

   mag #(
      .width_in_p(16),
      .width_out_p(16)
   ) mag_inst (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .valid_i(sobel_valid),
      .gx_i(gx),
      .gy_i(gy),
      .ready_o(sobel_ready),
      .valid_o(mag_valid),
      .mag_o(mag),
      .ready_i(mag_ready)
   );

   // HSV conversion
   wire [7:0] hsv_h, hsv_v;
   wire hsv_valid;
   wire hsv_ready;

   // Scale magnitude to 8-bit for HSV input with 16x boost
   wire [19:0] mag_scaled = {mag, 4'b0};  
   wire [7:0] mag_8bit = mag_scaled[11:4];  // Extract [11:4] for 8-bit output
   // Scale gradients to 8-bit (signed)
   wire [7:0] gx_8bit = gx[11:4];
   wire [7:0] gy_8bit = gy[11:4];

   hsv #(
      .width_p(8),
      .width_grad_p(8)
   ) hsv_inst (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .valid_i(mag_valid),
      .mag_i(mag_8bit),
      .gx_i(gx_8bit),
      .gy_i(gy_8bit),
      .ready_o(mag_ready),
      .valid_o(hsv_valid),
      .h_o(hsv_h),
      .v_o(hsv_v),
      .ready_i(hsv_ready)
   );

   // When hsv_valid is true, output H first, then V
   
   reg [7:0] hsv_h_r, hsv_v_r;
   reg hsv_has_h, hsv_has_v;
   
   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         hsv_h_r <= 8'h00;
         hsv_v_r <= 8'h00;
         hsv_has_h <= 1'b0;
         hsv_has_v <= 1'b0;
      end else begin
         // Store H,V when valid input arrives
         if (hsv_valid && !hsv_has_h) begin
            hsv_h_r <= hsv_h;
            hsv_v_r <= hsv_v;
            hsv_has_h <= 1'b1;
         end
         // Send H byte
         if (s_axis_uart_tready && s_axis_uart_tvalid && hsv_has_h && !hsv_has_v) begin
            hsv_has_v <= 1'b1;  // After H sent, prepare to send V
         end
         // Both bytes sent
         if (s_axis_uart_tready && s_axis_uart_tvalid && hsv_has_v) begin
            hsv_has_h <= 1'b0;
            hsv_has_v <= 1'b0;
         end
      end
   end
   
   // Output: if we have H stored, send H; if we have V, send V
   wire send_h = hsv_has_h && !hsv_has_v;
   wire send_v = hsv_has_v;
   
   assign s_axis_uart_tdata = send_h ? hsv_h_r : (send_v ? hsv_v_r : 8'h00);
   assign s_axis_uart_tvalid = hsv_has_h || hsv_has_v;
   assign hsv_ready = !hsv_has_h;  // Ready for new pixel when not storing

   assign led_o = {hsv_valid, tx_busy, rx_busy, rx_overrun_error, rx_frame_error};

endmodule
