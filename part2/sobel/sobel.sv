module sobel
  #(
    parameter width_in_p = 8,
    parameter width_out_p = 16,
    parameter linewidth_px_p = 480
   )
  (input [0:0] clk_i
  ,input [0:0] reset_i
  ,input [0:0] valid_i
  ,output [0:0] ready_o
  ,input [width_in_p - 1 : 0] data_i
  ,output logic signed [width_in_p+3:0] gx_o
  ,output logic signed [width_in_p+3:0] gy_o
  ,output [0:0] valid_o
  ,input [0:0] ready_i
  ,output [width_out_p - 1 : 0] data_o
  );

  logic [(2 * width_in_p) - 1 : 0] conv_data_o;
  logic conv_valid_o;
  logic conv_ready_i;
  
  wire [width_in_p-1:0] kernel [2:0][2:0];

  conv2d #(
    .linewidth_px_p(linewidth_px_p),
    .width_p(width_in_p)
  ) conv_inst (
    .clk_i(clk_i),
    .reset_i(reset_i),
    .valid_i(valid_i),
    .ready_o(ready_o),
    .data_i(data_i),
    .valid_o(conv_valid_o),
    .ready_i(conv_ready_i),
    .data_o(conv_data_o),
    .kernel(kernel)
  );

  // widen gradients to hold signed sums
  logic signed [width_in_p+3:0] gx, gy;
  logic signed [width_out_p - 1:0] magnitude;
  // temp absolute value signals
  logic signed [width_in_p+3:0] abs_gx, abs_gy;

  always_comb begin
    gx = -$signed({1'b0, kernel[0][0]}) + $signed({1'b0, kernel[0][2]})
         - 2 * $signed({1'b0, kernel[1][0]}) + 2 * $signed({1'b0, kernel[1][2]})
         - $signed({1'b0, kernel[2][0]}) + $signed({1'b0, kernel[2][2]});
    gy = -$signed({1'b0, kernel[0][0]}) - 2 * $signed({1'b0, kernel[0][1]}) - $signed({1'b0, kernel[0][2]})
         + $signed({1'b0, kernel[2][0]}) + 2 * $signed({1'b0, kernel[2][1]}) + $signed({1'b0, kernel[2][2]});
    if (gx < 0)
      abs_gx = -gx;
    else
      abs_gx = gx;
    if (gy < 0)
      abs_gy = -gy;
    else
      abs_gy = gy;
    magnitude = abs_gx + abs_gy;
    gx_o = gx;
    gy_o = gy;
  end

  logic [width_out_p - 1 : 0] data_r;
  logic valid_r;
  
  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      valid_r <= 1'b0;
      data_r <= '0;
    end else begin
      if (conv_ready_i) begin
        valid_r <= conv_valid_o;
        if (conv_valid_o) begin
          if (magnitude > ((1 << width_out_p) - 1))
            data_r <= (1 << width_out_p) - 1;
          else if (magnitude < 0)
            data_r <= '0;
          else
            data_r <= magnitude[width_out_p - 1 : 0];
        end
      end
    end
  end

  assign conv_ready_i = ~valid_r | ready_i;
  assign valid_o = valid_r;
  assign data_o = data_r;

endmodule