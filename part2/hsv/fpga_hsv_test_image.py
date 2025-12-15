#!/usr/bin/env python3
"""
FPGA HSV Image Processor

Loads an image, resizes to 480 pixels wide to match FPGA linewidth_px_p,
sends RGB pixels to FPGA via UART (which converts RGB→Gray→Sobel→HSV),
receives H, V output, reconstructs as full HSV with S=255, and saves output.

Usage:
python3 fpga_hsv_test_image.py --image [inputimage.jpg] --port [portname] --output [outputimage.jpg] --chunk-rows [numberofrows] --timeout [seconds]
    
Protocol:
- Send: RGB pixels (3 bytes per pixel: R, G, B) in 4096-byte chunks
- Receive: H, V output (2 bytes per pixel)
- Pipeline: RGB → RGB2Gray → Sobel → HSV → Output
    
Chunking:
For ice40 use --chunk-rows 8
"""

import cv2
import numpy as np
import argparse
import sys
import time
import os
from pathlib import Path

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

def load_image(image_path, as_rgb=True):
    """Load image in RGB or grayscale format."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    
    if as_rgb:
        # Convert BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    else:
        # Return grayscale
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_to_width(img, width=480):
    """Resize image to specified width, maintaining aspect ratio."""
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h))

def compute_sobel_gradients(gray):
    """Compute Sobel gradients and magnitude."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute magnitude 
    mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize to 8-bit range [0, 255]
    gx_norm = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gy_norm = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    mag_norm = np.clip(mag, 0, 255).astype(np.uint8)
    
    return gx_norm, gy_norm, mag_norm

def arctan2_approx(gy, gx):
    """
    Software implementation of arctan2 approximation used in FPGA.
    Maps angle to 0-180 degrees.
    FPGA formula: base_angle = (|gy| * 90) / (|gx| + |gy|)
    Then map to 0-180 based on sign of gx.
    """
    abs_gx = np.abs(gx.astype(np.float32))
    abs_gy = np.abs(gy.astype(np.float32))
    
    denom = abs_gx + abs_gy
    denom[denom == 0] = 1  # Avoid division by zero
    
    base_angle = (abs_gy * 90.0) / denom
    
    # Map to 0-180 based on gx sign
    angle = np.where(gx >= 0, base_angle, 180.0 - base_angle)
    
    return angle.astype(np.uint8)

def sigmoid_lut(mag_values):
    """
    Apply sigmoid LUT as used in FPGA.
    Function: f(x) = 256 / (1 + exp(-x/32))
    Precomputed for 8-bit input (0-255).
    I realized I could have just used the function, but I already made it for the sw test lmaoooo
    """
    sigmoid_table = np.array([
        0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x07, 0x07, 0x07, 0x07,
        0x07, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0b, 0x0c,
        0x0c, 0x0c, 0x0d, 0x0d, 0x0e, 0x0e, 0x0e, 0x0f, 0x0f, 0x10, 0x10, 0x11, 0x11, 0x12, 0x12, 0x13,
        0x13, 0x14, 0x14, 0x15, 0x16, 0x16, 0x17, 0x18, 0x18, 0x19, 0x1a, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d,
        0x2f, 0x30, 0x31, 0x32, 0x33, 0x35, 0x36, 0x37, 0x39, 0x3a, 0x3c, 0x3d, 0x3e, 0x40, 0x41, 0x43,
        0x45, 0x46, 0x48, 0x49, 0x4b, 0x4d, 0x4e, 0x50, 0x52, 0x54, 0x55, 0x57, 0x59, 0x5b, 0x5d, 0x5e,
        0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e,
        0x80, 0x81, 0x83, 0x85, 0x87, 0x89, 0x8b, 0x8d, 0x8f, 0x91, 0x93, 0x95, 0x97, 0x99, 0x9b, 0x9d,
        0x9f, 0xa1, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xab, 0xad, 0xaf, 0xb1, 0xb2, 0xb4, 0xb6, 0xb7, 0xb9,
        0xba, 0xbc, 0xbe, 0xbf, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc8, 0xc9, 0xca, 0xcc, 0xcd, 0xce, 0xcf,
        0xd0, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe0,
        0xe1, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe5, 0xe6, 0xe7, 0xe7, 0xe8, 0xe9, 0xe9, 0xea, 0xeb, 0xeb,
        0xec, 0xec, 0xed, 0xed, 0xee, 0xee, 0xef, 0xef, 0xf0, 0xf0, 0xf1, 0xf1, 0xf1, 0xf2, 0xf2, 0xf3,
        0xf3, 0xf3, 0xf4, 0xf4, 0xf4, 0xf5, 0xf5, 0xf5, 0xf5, 0xf6, 0xf6, 0xf6, 0xf7, 0xf7, 0xf7, 0xf7,
        0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf9, 0xf9, 0xf9, 0xf9, 0xf9, 0xf9, 0xfa, 0xfa, 0xfa, 0xfa, 0xfa
    ], dtype=np.uint8)
    
    indices = np.clip(mag_values.astype(int), 0, 255)
    return sigmoid_table[indices]

def convert_to_hsv_software(gx, gy, mag):
    """
    Software reference: convert using arctan2 approximation and sigmoid.
    Returns H, S, V arrays matching FPGA behavior.
    """
    h = arctan2_approx(gy, gx)
    v = sigmoid_lut(mag)
    s = np.full_like(h, 255)  # S is constant at 255
    
    return h, s, v

def create_hsv_visualization(h, s, v):
    """
    Create an HSV image for visualization.
    Returns RGB image with HSV values mapped to colors.
    OpenCV format: H in 0-180, S in 0-255, V in 0-255.
    """
    hsv_img = np.dstack((h, s, v)).astype(np.uint8)
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return rgb_img

def send_image_to_fpga(ser, rgb_img, chunk_rows=4, timeout=120, live_display=False):
    """
    Send RGB image to FPGA and receive HSV output.
    Protocol:
    - Send: RGB pixels in 4096-byte chunks
    - Receive: H, V output (480×w pixels @ 2 bytes each)
    live_display: Enable real-time visualization of output image
    """
    height, width = rgb_img.shape[:2]
    
    if width != 480:
        print(f"WARNING: Image width is {width}, expected 480. Resizing...")
        rgb_img = resize_to_width(rgb_img, 480)
        height, width = rgb_img.shape[:2]
    
    # Allocate output array
    h_fpga = np.zeros((height, width), dtype=np.uint8)
    v_fpga = np.zeros((height, width), dtype=np.uint8)
    
    # Setup live display if enabled
    display_available = False
    frames_dir = None
    
    if live_display:
        # Check if display is available (X11 or Wayland)
        has_display = bool(os.environ.get('DISPLAY')) or bool(os.environ.get('WAYLAND_DISPLAY'))
        
        if has_display:
            # Try to create OpenCV window
            try:
                # Create initial HSV visualization
                s_temp = np.full((height, width), 255, dtype=np.uint8)
                hsv_temp = np.dstack((h_fpga, s_temp, v_fpga))
                rgb_temp = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2RGB)
                
                cv2.namedWindow('FPGA HSV Output', cv2.WINDOW_NORMAL)
                cv2.imshow('FPGA HSV Output', cv2.cvtColor(rgb_temp, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                display_available = True
                print("Live display enabled (OpenCV window)")
            except Exception as e:
                print(f"Live display: Failed to create window ({e}), saving frames to ./frames/")
                frames_dir = Path("frames")
                frames_dir.mkdir(exist_ok=True)
                display_available = False
        else:
            print("Live display: No display server detected, saving frames to ./frames/")
            frames_dir = Path("frames")
            frames_dir.mkdir(exist_ok=True)
            display_available = False
    
    # Process in chunks
    num_chunks = (height + chunk_rows - 1) // chunk_rows
    
    for chunk_idx in range(num_chunks):
        start_row = chunk_idx * chunk_rows
        end_row = min(start_row + chunk_rows, height)
        chunk_h = end_row - start_row
        
        print(f"\nChunk {chunk_idx+1}/{num_chunks}: rows {start_row}-{end_row-1} ({chunk_h} rows) ---")
        
        # Extract chunk
        chunk_rgb = rgb_img[start_row:end_row, :, :]
        
        # Send 24-bit RGB (3 bytes per pixel) 
        chunk_rgb_bytes = chunk_rgb.astype(np.uint8).tobytes()
        
        tx_bytes = chunk_rgb_bytes
        expected_rx = chunk_h * width * 2  # 2 bytes per pixel (H, V)
        
        print(f"  Sending {chunk_h*width} pixels ({width}×{chunk_h}) = {len(tx_bytes)} bytes")
        
        # Send in larger chunks for better throughput (shoutout cse120)
        ser.reset_input_buffer()
        CHUNK_SIZE = 4096  
        written = 0
        for i in range(0, len(tx_bytes), CHUNK_SIZE):
            chunk = tx_bytes[i:i+CHUNK_SIZE]
            written += ser.write(chunk)
            ser.flush()
            time.sleep(0.0001)  
        
        print(f"  Sent {written}/{len(tx_bytes)} bytes")
        
        # Sobel needs ~480 pixels buffer = ~2-3 rows
        # RGB2Gray: 1 cycle, MAG: 2 cycles, HSV: 2 cycles
        if chunk_idx == 0:
            pipeline_delay = 0.5  
        else:
            pipeline_delay = 0.2 
        pipeline_delay += chunk_h * 0.003  
        print(f"  Waiting {pipeline_delay:.2f}s for pipeline latency...")
        time.sleep(pipeline_delay)
        
        # Receive output
        print(f"  Receiving HSV data (expecting {expected_rx} bytes)...")
        rx_buffer = bytearray()
        start_time = time.time()
        last_log = 0
        no_data_count = 0
        max_no_data = 20  
        
        while (time.time() - start_time) < timeout:
            try:
                data = ser.read(min(4096, expected_rx - len(rx_buffer)))
                if data:
                    rx_buffer.extend(data)
                    no_data_count = 0
                    if len(rx_buffer) - last_log >= 4096:
                        progress_pct = 100 * len(rx_buffer) / expected_rx
                        print(f"  Received {len(rx_buffer)}/{expected_rx} bytes ({progress_pct:.1f}%)")
                        last_log = len(rx_buffer)
                    if len(rx_buffer) >= expected_rx:
                        break
                else:
                    no_data_count += 1
                    if no_data_count > max_no_data:
                        break
            except Exception as e:
                print(f"  Read error: {e}")
                break
            time.sleep(0.001)  
        
        rx_dt = time.time() - start_time
        print(f"  Received {len(rx_buffer)} bytes in {rx_dt:.3f}s")
        
        if len(rx_buffer) == 0:
            print(f"ERROR: No data received for chunk {chunk_idx+1}")
            print(f"FPGA may not be responding. Check:")
            print(f"1. FPGA is programmed with correct bitstream")
            print(f"2. Serial port connection is solid")
            print(f"3. FPGA clock is running (check LED blinks)")
            ser.close()
            return None
        
        # Parse H, V values (2 bytes per pixel)
        if len(rx_buffer) >= expected_rx:
            for row in range(chunk_h):
                for col in range(width):
                    pixel_idx = row * width + col
                    byte_idx = pixel_idx * 2
                    
                    h_val = rx_buffer[byte_idx]
                    v_val = rx_buffer[byte_idx + 1]
                    
                    h_fpga[start_row + row, col] = h_val
                    v_fpga[start_row + row, col] = v_val
        else:
            print(f"  WARNING: Incomplete data received ({len(rx_buffer)}/{expected_rx})")
            for i in range(0, len(rx_buffer), 2):
                pixel_idx = i // 2
                row = (start_row * width + pixel_idx) // width
                col = (start_row * width + pixel_idx) % width
                
                if row < height and pixel_idx < len(rx_buffer) // 2:
                    h_fpga[row, col] = rx_buffer[i]
                    if i + 1 < len(rx_buffer):
                        v_fpga[row, col] = rx_buffer[i + 1]
        
        # Report chunk statistics
        h_range = h_fpga[start_row:end_row].max()
        v_range = (v_fpga[start_row:end_row]).max()
        v_mean = (v_fpga[start_row:end_row]).mean()
        print(f"  Chunk complete: H range [0, {h_range}], V range [0, {v_range}], V mean {v_mean:.1f}")
        
        # Update live display
        if live_display:
            progress_pct = 100*(chunk_idx+1)/num_chunks
            
            # Create HSV visualization for current state
            s_display = np.full((height, width), 255, dtype=np.uint8)
            hsv_display = np.dstack((h_fpga, s_display, v_fpga))
            rgb_display = cv2.cvtColor(hsv_display, cv2.COLOR_HSV2RGB)
            
            if display_available:
                # Update OpenCV window
                cv2.imshow('FPGA HSV Output', cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                print(f"  → Display updated ({progress_pct:.0f}% complete)")
            elif frames_dir:
                # Save frame to directory
                frame_path = frames_dir / f"frame_{chunk_idx+1:04d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR))
                print(f"  → Saved: {frame_path} ({progress_pct:.0f}% complete)")
    
    # Finalize live display
    if live_display:
        if display_available:
            print("\n✓ Live window updated - Press any key in the window to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif frames_dir:
            print(f"\n✓ Frames saved to: {frames_dir}/")
            print(f"  View with: ls {frames_dir}/ | xargs -I {{}} code {frames_dir}/{{}}")
    
    return h_fpga, v_fpga

def main():
    parser = argparse.ArgumentParser(description="FPGA HSV Test - Process Image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--port", default=None, help="Serial port (e.g., /dev/ttyUSB0 or COM3). If not provided, runs software reference only.")
    parser.add_argument("--output", help="Output filename (default: hsv_<input>.jpg)")
    parser.add_argument("--timeout", type=float, default=120, help="Read timeout in seconds (default: 120)")
    parser.add_argument("--width", type=int, default=480, help="Resize to width (default: 480)")
    parser.add_argument("--chunk-rows", type=int, default=4, help="Rows per chunk (default: 4)")
    parser.add_argument("--no-compare", action="store_true", help="Skip software comparison")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--live-display", action='store_true', 
                        help="Display output image in real-time as it's being processed")
    
    args = parser.parse_args()
    
    if not SERIAL_AVAILABLE:
        print("ERROR: pyserial not installed. pip install pyserial")
        return 1
    
    try:
        # Load and resize image
        print(f"Loading image: {args.image}")
        rgb_img = load_image(args.image, as_rgb=True)
        height, width = rgb_img.shape[:2]
        print(f"Original size: {width}×{height}")
        
        rgb_img = resize_to_width(rgb_img, args.width)
        height, width = rgb_img.shape[:2]
        print(f"Resized to: {width}×{height} RGB")
        
        # Software reference
        if not args.no_compare:
            print(f"\n=== SOFTWARE REFERENCE ===")
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            print("Converted to grayscale")
            
            gx, gy, mag = compute_sobel_gradients(gray)
            print(f"Gradient range: Gx [{gx.min()}, {gx.max()}], Gy [{gy.min()}, {gy.max()}]")
            print(f"Magnitude range: [{mag.min()}, {mag.max()}]")
            
            h_sw, s_sw, v_sw = convert_to_hsv_software(gx, gy, mag)
            hsv_sw = create_hsv_visualization(h_sw, s_sw, v_sw)
            
            print(f"H range: {h_sw.min()}-{h_sw.max()}")
            print(f"V range: {v_sw.min()}-{v_sw.max()}")
            print(f"V mean: {v_sw.mean():.1f}")
        else:
            h_sw = s_sw = v_sw = hsv_sw = None
        
        # FPGA test
        print(f"\n=== FPGA TEST ===")
        if args.port is None:
            print("No serial port specified. Skipping FPGA test.")
            h_fpga = v_fpga = hsv_fpga = None
        else:
            try:
                ser = serial.Serial(args.port, baudrate=args.baud, timeout=0.5)
                time.sleep(0.2)
            except Exception as e:
                print(f"ERROR: Cannot open serial port {args.port}: {e}")
                print(f"ERROR: Failed to get FPGA data")
                return 1
            
            result = send_image_to_fpga(ser, rgb_img, chunk_rows=args.chunk_rows, timeout=args.timeout, live_display=args.live_display)
            ser.close()
            
            if result is None:
                return 1
            
            h_fpga, v_fpga = result
            
            print(f"\nFPGA H range: {h_fpga.min()}-{h_fpga.max()}")
            print(f"FPGA V range: {v_fpga.min()}-{v_fpga.max()}")
            print(f"FPGA V mean: {v_fpga.mean():.1f}")
            
            # Reconstruct full HSV with S=255
            s_fpga = np.full_like(h_fpga, 255)
            hsv_fpga = create_hsv_visualization(h_fpga, s_fpga, v_fpga)
        
        # Comparison
        if not args.no_compare and h_sw is not None and h_fpga is not None:
            print(f"\n=== COMPARISON ===")
            if h_sw.shape == h_fpga.shape:
                h_diff = np.abs(h_sw.astype(np.int32) - h_fpga.astype(np.int32))
                v_diff = np.abs(v_sw.astype(np.int32) - v_fpga.astype(np.int32))
                
                print(f"H - Max difference: {h_diff.max()}")
                print(f"H - Mean difference: {h_diff.mean():.2f}")
                print(f"V - Max difference: {v_diff.max()}")
                print(f"V - Mean difference: {v_diff.mean():.2f}")
                
                h_match_pct = 100 * np.sum(h_diff == 0) / h_diff.size
                h_within_5_pct = 100 * np.sum(h_diff <= 5) / h_diff.size
                v_match_pct = 100 * np.sum(v_diff == 0) / v_diff.size
                v_within_5_pct = 100 * np.sum(v_diff <= 5) / v_diff.size
                
                print(f"H - Perfect match: {h_match_pct:.1f}%, Within 5: {h_within_5_pct:.1f}%")
                print(f"V - Perfect match: {v_match_pct:.1f}%, Within 5: {v_within_5_pct:.1f}%")
                
                if h_within_5_pct >= 10 and v_within_5_pct >= 30:
                    print("✓ PASS: Good agreement between FPGA and software reference (hardware-expected)")
                elif h_within_5_pct >= 5 and v_within_5_pct >= 20:
                    print("⚠ WARN: Moderate agreement - minor adjustments may help")
                else:
                    print("✗ FAIL: Poor agreement - verify FPGA design")
            else:
                print(f"ERROR: Shape mismatch - SW: {h_sw.shape}, FPGA: {h_fpga.shape}")
        
        # Save outputs
        print(f"\n=== SAVING OUTPUTS ===")
        
        if args.output:
            output_fpga = args.output if h_fpga is not None else None
            base = Path(args.output).stem
            output_sw = f"{base}_sw.jpg" if not args.no_compare else None
            output_input = f"{base}_input.jpg"
        else:
            base = Path(args.image).stem
            output_sw = f"hsv_sw_{base}.jpg" if not args.no_compare else None
            output_fpga = f"hsv_fpga_{base}.jpg" if h_fpga is not None else None
            output_input = f"input_{base}.jpg"
        
        # Save input image
        cv2.imwrite(output_input, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        print(f"  Input: {output_input}")
        
        # Save FPGA output
        if h_fpga is not None:
            cv2.imwrite(output_fpga, cv2.cvtColor(hsv_fpga, cv2.COLOR_RGB2BGR))
            print(f"  FPGA output: {output_fpga}")
        
        # Save software reference if computed
        if not args.no_compare and hsv_sw is not None:
            cv2.imwrite(output_sw, cv2.cvtColor(hsv_sw, cv2.COLOR_RGB2BGR))
            print(f"  Software reference: {output_sw}")
        
        print(f"\n✓ Processing complete!")
        return 0
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
