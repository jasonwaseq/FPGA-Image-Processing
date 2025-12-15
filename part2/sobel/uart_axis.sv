module uart_axis
  #(parameter linewidth_px_p = 480)  // Image width in pixels
  (input [0:0] clk_i
  ,input [0:0] reset_i
  ,input [0:0] rx_serial_i
  ,output [0:0] tx_serial_o
  ,output [5:1] led_o
   );

   localparam [31:0] data_width_lp = 8;
   localparam width_out_lp = 16;
   
   wire [data_width_lp-1:0] m_axis_uart_tdata;
   wire       m_axis_uart_tvalid;
   wire       m_axis_uart_tready;

   wire [data_width_lp-1:0] s_axis_uart_tdata;
   wire       s_axis_uart_tvalid;
   wire       s_axis_uart_tready;

   wire [31:0] m_axis_tdata;
   wire        m_axis_tvalid;
   wire        m_axis_tlast;
   wire [3:0]  m_axis_tkeep;
   wire        m_axis_tready;
   
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
      .prescale(16'd27)
   );

   axis_adapter #(
      .S_DATA_WIDTH(data_width_lp),
      .M_DATA_WIDTH(32),
      .S_KEEP_ENABLE(0),
      .M_KEEP_ENABLE(1),
      .M_KEEP_WIDTH(4),
      .ID_ENABLE(0),
      .DEST_ENABLE(0),
      .USER_ENABLE(0)
   ) widener (
      .clk(clk_i),
      .rst(reset_i),
      .s_axis_tdata(m_axis_uart_tdata),
      .s_axis_tkeep(1'b1),
      .s_axis_tvalid(m_axis_uart_tvalid),
      .s_axis_tready(m_axis_uart_tready),
      .s_axis_tlast(1'b0),  
      .s_axis_tid(),
      .s_axis_tdest(),
      .s_axis_tuser(),

      .m_axis_tdata(m_axis_tdata),
      .m_axis_tkeep(m_axis_tkeep),
      .m_axis_tvalid(m_axis_tvalid),
      .m_axis_tready(m_axis_tready),
      .m_axis_tlast(m_axis_tlast),
      .m_axis_tid(),
      .m_axis_tdest(),
      .m_axis_tuser()
   );

   logic gray_valid;
   logic [data_width_lp-1:0] gray_data;
   logic gray_ready;
   
   rgb2gray #(
      .width_p(data_width_lp)
   ) rgb2gray_inst (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .valid_i(m_axis_tvalid),
      .red_i(m_axis_tdata[7:0]),
      .green_i(m_axis_tdata[15:8]),
      .blue_i(m_axis_tdata[23:16]),
      .ready_o(m_axis_tready),
      .valid_o(gray_valid),
      .gray_o(gray_data),
      .ready_i(gray_ready)
   );

   // sobel edge detection
   logic sobel_valid;
   logic [width_out_lp-1:0] sobel_data;
   logic sobel_ready;
   
   sobel #(
      .width_in_p(data_width_lp),
      .width_out_p(width_out_lp),
      .linewidth_px_p(linewidth_px_p)
   ) sobel_inst (
      .clk_i(clk_i),
      .reset_i(reset_i),
      .valid_i(gray_valid),
      .data_i(gray_data),
      .ready_o(gray_ready),
      .valid_o(sobel_valid),
      .data_o(sobel_data),
      .ready_i(sobel_ready)
   );

   // Scale down from 16-bits to 8-bits
   logic [data_width_lp-1:0] sobel_scaled;
   assign sobel_scaled = sobel_data[15:8];  // take upper 8 bits
   
   // Convert back to 32-bits for narrower adapter
   logic [31:0] sobel_extended;
   assign sobel_extended = {8'h00, sobel_scaled, sobel_scaled, sobel_scaled};
   
   axis_adapter #(
      .S_DATA_WIDTH(32),
      .M_DATA_WIDTH(data_width_lp),
      .S_KEEP_ENABLE(1),
      .S_KEEP_WIDTH(4),
      .M_KEEP_ENABLE(0),
      .ID_ENABLE(0),
      .DEST_ENABLE(0),
      .USER_ENABLE(0)
   ) narrower (
      .clk(clk_i),
      .rst(reset_i),
      .s_axis_tdata(s_axis_tdata),
      .s_axis_tkeep(s_axis_tkeep),
      .s_axis_tvalid(s_axis_tvalid),
      .s_axis_tready(s_axis_tready),
      .s_axis_tlast(s_axis_tlast),
      .s_axis_tid(),
      .s_axis_tdest(),
      .s_axis_tuser(),

      .m_axis_tdata(s_axis_uart_tdata),
      .m_axis_tkeep(),
      .m_axis_tvalid(s_axis_uart_tvalid),
      .m_axis_tready(s_axis_uart_tready),
      .m_axis_tlast(),
      .m_axis_tid(),
      .m_axis_tdest(),
      .m_axis_tuser()
   );
   
   assign s_axis_tdata = sobel_extended;
   assign s_axis_tvalid = sobel_valid;
   assign s_axis_tkeep = 4'b1111;
   assign s_axis_tlast = 1'b0;
   assign sobel_ready = s_axis_tready;
   
   assign led_o = 5'b10101; 
   
endmodule
