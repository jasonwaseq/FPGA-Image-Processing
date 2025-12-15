module rgb2gray
  #(
    parameter width_p = 8
   )
  (
    input  logic             clk_i
   ,input  logic             reset_i

   ,input  logic             valid_i
   ,input  logic [width_p-1:0] red_i
   ,input  logic [width_p-1:0] green_i
   ,input  logic [width_p-1:0] blue_i
   ,output logic             ready_o

   ,output logic             valid_o
   ,output logic [width_p-1:0] gray_o
   ,input  logic             ready_i
  );

  localparam [15:0] red = 16'd19595;   // 0.2989
  localparam [15:0] green = 16'd38470; // 0.5870
  localparam [15:0] blue = 16'd7471;   // 0.1140

  localparam prod_width_lp = width_p + 16;

  logic [(prod_width_lp+1):0] r_prod, g_prod, b_prod;
  
  logic [(prod_width_lp+1):0] gray_sum;

  always_comb begin
    r_prod   = {2'b00, red_i}   * red;
    g_prod   = {2'b00, green_i} * green;
    b_prod   = {2'b00, blue_i}  * blue;
    gray_sum = r_prod + g_prod + b_prod;
  end

  wire [width_p-1:0] gray_r;
  logic [prod_width_lp+1:0] gray_shifted;
  logic valid_r;

  always_ff @(posedge clk_i) begin
  if (reset_i) begin
    valid_r <= 1'b0;
  end else begin
    if (ready_o && valid_i) begin
      gray_shifted <= gray_sum >> 16;
      valid_r <= 1'b1;
    end
    else if (ready_i && valid_r) begin
      valid_r <= 1'b0; 
    end
  end
  end
  
  assign gray_r = gray_shifted[width_p-1:0];

  assign ready_o = ready_i | ~valid_r;
  assign valid_o = valid_r;
  assign gray_o  = gray_r[width_p-1:0];

endmodule
