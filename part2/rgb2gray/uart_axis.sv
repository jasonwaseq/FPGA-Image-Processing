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

   wire        rgb2gray_valid;
   wire [7:0]  rgb2gray_gray;
   wire        rgb2gray_ready;

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
      .s_axis_tlast(0),  
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
   
   rgb2gray #(
      .width_p(8)
   ) rgb2gray_inst (
      .clk_i(clk_i),
      .reset_i(reset_i),
      
      .valid_i(m_axis_tvalid),
      .red_i(m_axis_tdata[7:0]),      // Byte 0: Red
      .green_i(m_axis_tdata[15:8]),   // Byte 1: Green
      .blue_i(m_axis_tdata[23:16]),   // Byte 2: Blue
      .ready_o(m_axis_tready),
      
      .valid_o(rgb2gray_valid),
      .gray_o(rgb2gray_gray),
      .ready_i(rgb2gray_ready)
   );
   
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
   
   assign s_axis_tdata = {m_axis_tdata[31:24], rgb2gray_gray, rgb2gray_gray, rgb2gray_gray};
   assign s_axis_tvalid = rgb2gray_valid;
   assign s_axis_tkeep = 4'b1111;
   assign s_axis_tlast = m_axis_tlast;
   assign rgb2gray_ready = s_axis_tready;
   
endmodule
