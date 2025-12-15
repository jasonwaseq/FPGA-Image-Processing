// Top-level design file for the icebreaker FPGA board
module top
  (input [0:0] clk_12mhz_i
  // n: Negative Polarity (0 when pressed, 1 otherwise)
  // async: Not synchronized to clock
  // unsafe: Not De-Bounced
  ,input [0:0] reset_n_async_unsafe_i
  // async: Not synchronized to clock
  // unsafe: Not De-Bounced
  ,input [3:1] button_async_unsafe_i

  // Line Out (Green)
  // Main clock (for synchronization)
  ,output tx_main_clk_o
  // Selects between L/R channels, but called a "clock"
  ,output tx_lr_clk_o
  // Data clock
  ,output tx_data_clk_o
  // Output Data
  ,output tx_data_o

  // Line In (Blue)
  // Main clock (for synchronization)
  ,output rx_main_clk_o
  // Selects between L/R channels, but called a "clock"
  ,output rx_lr_clk_o
  // Data clock
  ,output rx_data_clk_o
  // Input data
  ,input  rx_data_i

  ,output [5:1] led_o);

   wire        clk_o;

   // These two D Flip Flops form what is known as a Synchronizer. We
   // will learn about these in Week 5, but you can see more here:
   // https://inst.eecs.berkeley.edu/~cs150/sp12/agenda/lec/lec16-synch.pdf
   wire reset_n_sync_r;
   wire reset_sync_r;
   wire reset_r; // Use this as your reset_signal

   dff
     #()
   sync_a
     (.clk_i(clk_o)
     ,.reset_i(1'b0)
     ,.en_i(1'b1)
     ,.d_i(reset_n_async_unsafe_i)
     ,.q_o(reset_n_sync_r));

   inv
     #()
   inv
     (.a_i(reset_n_sync_r)
     ,.b_o(reset_sync_r));

   dff
     #()
   sync_b
     (.clk_i(clk_o)
     ,.reset_i(1'b0)
     ,.en_i(1'b1)
     ,.d_i(reset_sync_r)
     ,.q_o(reset_r));
       
   wire [31:0] axis_tx_data;
   wire        axis_tx_valid;
   wire        axis_tx_ready;
   wire        axis_tx_last;
   
   wire [31:0] axis_rx_data;
   wire        axis_rx_valid;
   wire        axis_rx_ready;
   wire        axis_rx_last;

  (* blackbox *)
  // This is a PLL! You'll learn about these later...
  SB_PLL40_PAD 
    #(.FEEDBACK_PATH("SIMPLE")
     ,.PLLOUT_SELECT("GENCLK")
     ,.DIVR(4'b0000)
     ,.DIVF(7'b1000011)
     ,.DIVQ(3'b101)
     ,.FILTER_RANGE(3'b001)
     )
   pll_inst
     (.PACKAGEPIN(clk_12mhz_i)
     ,.PLLOUTCORE(clk_o)
     ,.RESETB(1'b1)
     ,.BYPASS(1'b0)
     );
  
   assign axis_clk = clk_o;

   assign axis_tx_data[31:24] = 8'b0;
   axis_i2s2 
     #()
   i2s2_inst
     (.axis_clk(axis_clk)
     ,.axis_resetn(~reset_r)
      
     ,.tx_axis_c_data(axis_tx_data)
     ,.tx_axis_c_valid(axis_tx_valid)
     ,.tx_axis_c_ready(axis_tx_ready)
     ,.tx_axis_c_last(axis_tx_last)
     
     ,.rx_axis_p_data(axis_rx_data)
     ,.rx_axis_p_valid(axis_rx_valid)
     ,.rx_axis_p_ready(axis_rx_ready)
     ,.rx_axis_p_last(axis_rx_last)
     
     ,.tx_mclk(tx_main_clk_o)
     ,.tx_lrck(tx_lr_clk_o)
     ,.tx_sclk(tx_data_clk_o)
     ,.tx_sdout(tx_data_o)
     ,.rx_mclk(rx_main_clk_o)
     ,.rx_lrck(rx_lr_clk_o)
     ,.rx_sclk(rx_data_clk_o)
     ,.rx_sdin(rx_data_i)
     );


/*   assign axis_tx_data = axis_rx_data;
   assign axis_tx_last = axis_rx_last;
   assign axis_tx_valid = axis_rx_valid;
   assign axis_rx_ready = axis_tx_ready;
   assign axis_tx_data = axis_rx_data;
*/
   // Input Interface (l for local)
   wire [0:0]        valid_li;
   wire [0:0]        ready_lo;

   wire [23:0] data_right_li;
   wire [23:0] data_left_li;

   // Output Interface (l for local)
   wire [0:0]        valid_lo;
   wire [0:0]        ready_li;        

   wire [23:0] data_right_lo;
   wire [23:0] data_left_lo;

   // Serial in, Parallel out
   sipo
    #()
   sipo_inst
     (.clk_i                            (clk_o)
     ,.reset_i                          (reset_r)
      // Outputs (Input Interface to your module)
     ,.\data_o[1]                       (data_right_li)
     ,.\data_o[0]                       (data_left_li)
     ,.v_o                              (valid_li)
     ,.ready_i                          (ready_lo & valid_li)
     // Inputs (Don't worry about these)
     ,.ready_and_o                      (axis_rx_ready)
     ,.data_i                           (axis_rx_data[23:0])
     ,.v_i                              (axis_rx_valid)
     );

   // Parallel in, Serial out
   piso
    #()
   piso_inst
     (.clk_i                            (clk_o)
     ,.reset_i                          (reset_r)
     // Outputs (Don't worry about these)
     // Use the low-order bit to signal last
     ,.data_o                           ({axis_tx_data[23:0], axis_tx_last})
     ,.valid_o                          (axis_tx_valid)
     ,.ready_i                          (axis_tx_ready)
     // Inputs (Output interface from your module)
     ,.\data_i[1]                       ({data_right_lo, 1'b1})
     ,.\data_i[0]                       ({data_left_lo, 1'b0})
     ,.valid_i                          (valid_lo)
     ,.ready_and_o                      (ready_li)
     );

   conv1d
     #()
     conv_inst
       (// Outputs
        .ready_o                        (ready_lo),
        .valid_o                        (valid_lo),
        .data_o                         (data_right_lo),
        // Inputs
        .clk_i                          (clk_o),
        .reset_i                        (reset_r),
        .valid_i                        (valid_li),
        .data_i                         (data_right_li),
        .ready_i                        (ready_li));

   // You should drive right_lo and left_lo
//   assign data_right_lo = 0;
   assign data_left_lo = 0;
                         
endmodule
