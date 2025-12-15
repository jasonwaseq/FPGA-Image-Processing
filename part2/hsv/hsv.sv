module hsv
  #(
   // This is here to help, but we won't change it.
   parameter width_p = 8
  ,parameter width_grad_p = 8
   )
  (input [0:0] clk_i
  ,input [0:0] reset_i

  ,input [0:0] valid_i
   // You'll get mag_i from the magnitude module. The magnitude module
   // will also need to provide gx and gy_i to this module -- passing
   // them through unmodified.
  ,input [width_p - 1:0] mag_i
  ,input [width_grad_p - 1:0] gx_i
  ,input [width_grad_p - 1:0] gy_i
  ,output [0:0] ready_o

  ,output [0:0] valid_o
  ,output [width_p - 1:0] h_o
  ,output [width_p - 1:0] v_o
  ,input [0:0] ready_i
  );

  // Stage 1: Register inputs
  logic signed [width_grad_p-1:0] gx_r, gy_r;
  logic [width_p-1:0] mag_r;
  logic valid_s1;
  logic ready_s1;
  
  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      gx_r <= '0;
      gy_r <= '0;
      mag_r <= '0;
      valid_s1 <= 1'b0;
    end else if (ready_s1) begin
      if (valid_i) begin
        gx_r <= $signed(gx_i);
        gy_r <= $signed(gy_i);
        mag_r <= mag_i;
        valid_s1 <= 1'b1;
      end else begin
        valid_s1 <= 1'b0;
      end
    end
  end
  
  // Stage 2: Compute hue and value outputs
  logic [width_p-1:0] h_data, v_data;
  logic valid_s2;
  logic ready_s2;
  logic signed [width_grad_p:0] gx_abs, gy_abs;
  
  // Hue computation: simplified angle approximation
  always_comb begin    
    gx_abs = (gx_r < 0) ? -gx_r : gx_r;
    gy_abs = (gy_r < 0) ? -gy_r : gy_r;
    if (gx_abs == 0 && gy_abs == 0) begin
      h_data = 8'd0;
    end else if (gx_r >= 0 && gy_r >= 0) begin
      // Q1: 0-90
      if (gx_abs >= gy_abs) begin
        h_data = 8'd45;  // More horizontal
      end else begin
        h_data = 8'd0;   // More vertical
      end
    end else if (gx_r < 0 && gy_r >= 0) begin
      // Q2: 90-180
      if (gx_abs >= gy_abs) begin
        h_data = 8'd135; // More horizontal
      end else begin
        h_data = 8'd90;  // More vertical
      end
    end else if (gx_r < 0 && gy_r < 0) begin
      // Q3: 180
      h_data = 8'd180;
    end else begin
      // Q4: 0-90
      if (gx_abs >= gy_abs) begin
        h_data = 8'd45;  // More horizontal
      end else begin
        h_data = 8'd0;   // More vertical
      end
    end
  end
  
  // Value computation: apply sigmoid to magnitude
  always_comb begin
    // sigmoid(x) = linear response to magnitude
    if (mag_r < 16) begin
      v_data = mag_r;  // Linear: 0-16
    end else if (mag_r < 128) begin
      v_data = 16 + ((mag_r - 16) >> 1);  // 16-80 for input 16-128
    end else begin
      v_data = 80 + ((mag_r - 128) >> 1);  // 80-192 for input 128-255
    end
  end
  
  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      valid_s2 <= 1'b0;
    end else if (ready_s2) begin
      valid_s2 <= valid_s1;
    end
  end
  
  assign ready_s1 = ~valid_s1 | ready_s2;
  assign ready_s2 = ~valid_s2 | (~valid_o | ready_i);
  assign ready_o = ready_s1;
  
  assign valid_o = valid_s2;
  assign h_o = h_data;
  assign v_o = v_data;
   
endmodule
