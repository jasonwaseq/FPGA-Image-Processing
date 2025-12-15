module mag
  #(
   // This is here to help, but we won't change it.
   parameter width_in_p = 16
  ,parameter width_out_p = 16
   )
  (input [0:0] clk_i
  ,input [0:0] reset_i

  ,input [0:0] valid_i
  ,input [width_in_p - 1:0] gx_i
  ,input [width_in_p - 1:0] gy_i
  ,output [0:0] ready_o

  ,output [0:0] valid_o
  ,output [width_out_p - 1:0] mag_o
  ,input [0:0] ready_i
  );

  // Stage 1: Compute absolute values
  logic [width_in_p-1:0] abs_gx_r, abs_gy_r;
  logic valid_s1;
  logic ready_s1;
  logic signed [width_in_p-1:0] gx_s, gy_s;
  assign gx_s = $signed(gx_i);
  assign gy_s = $signed(gy_i);
  
  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      abs_gx_r <= '0;
      abs_gy_r <= '0;
      valid_s1 <= 1'b0;
    end else begin
      if (ready_s1) begin
        valid_s1 <= valid_i;
        if (valid_i) begin
          abs_gx_r <= (gx_s < 0) ? -gx_s : gx_s;
          abs_gy_r <= (gy_s < 0) ? -gy_s : gy_s;
        end
      end
    end
  end

  // Stage 2: Approximate magnitude calculation
  logic [width_in_p:0] mag_approx_r;
  logic valid_s2;
  logic ready_s2;
  logic [width_in_p:0] max_v, min_v;

  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      mag_approx_r <= '0;
      valid_s2 <= 1'b0;
    end else begin
      if (ready_s2) begin
        valid_s2 <= valid_s1;
        if (valid_s1) begin
          if (abs_gx_r >= abs_gy_r) begin
            max_v = {1'b0, abs_gx_r};
            min_v = {1'b0, abs_gy_r};
          end else begin
            max_v = {1'b0, abs_gy_r};
            min_v = {1'b0, abs_gx_r};
          end
          mag_approx_r <= max_v + (min_v >> 1); // Magnitude = max + (min >> 1)
        end
      end
    end
  end

  // Stage 3: Output
  logic [width_out_p-1:0] data_r;
  logic valid_r;

  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      valid_r <= 1'b0;
      data_r <= '0;
    end else begin
      if (ready_i || ~valid_r) begin
        valid_r <= valid_s2;
        if (valid_s2) begin
          if (mag_approx_r > ((1 << width_out_p) - 1))
            data_r <= {width_out_p{1'b1}};
          else
            data_r <= width_out_p'(mag_approx_r);
        end
      end
    end
  end

  // stage 1 can accept when stage 2 is ready or empty
  assign ready_s1 = ~valid_s1 | ready_s2;
  // stage 2 can accept when output stage is ready or empty  
  assign ready_s2 = ~valid_s2 | (~valid_r | ready_i);
  // Module ready when stage 1 is ready
  assign ready_o = ready_s1;
  
  assign valid_o = valid_r;
  assign mag_o = data_r;

endmodule
