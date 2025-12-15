#!/usr/bin/env python3
"""
FPGA Magnitude Image Processor

Loads an image, resizes to 480 pixels wide to match FPGA linewidth_px_p,
sends RGB pixels to FPGA via UART (which converts RGB→Gray→Sobel→Magnitude),
receives 8-bit magnitude output, and compares with software reference.

Usage:
python3 fpga_mag_test_image.py --image [inputimage.jpg] --port [portname] --output [outputimage.jpg] --timeout [seconds] --chunk-rows [rows]
    
Protocol:
- Send: RGB pixels (3 bytes per pixel, width=480 pixels)
- Receive: Grayscale magnitude output (1 byte per pixel, scaled from 16-bit)
- Pipeline: RGB → RGB2Gray → Sobel → Magnitude → Output
    
Chunking:
For ice40 FPGA --chunk rows 8 is good
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray


def resize_to_linewidth(img, linewidth=480):
    """Resize image to exact linewidth while preserving aspect ratio."""
    if len(img.shape) == 2:  # Grayscale
        h, w = img.shape
    else:  # RGB
        h, w, _ = img.shape
    
    new_w = linewidth
    new_h = int(h * new_w / w)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_resized

def rgb_to_gray_software(rgb):
    """Convert RGB to grayscale using same weights as FPGA rgb2gray module."""
    # FPGA uses: 0.2989*R + 0.5870*G + 0.1140*B
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def compute_sobel_gradients(gray):
    """Compute Sobel gradients matching FPGA 12-bit signed output."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # OpenCV Sobel with 8-bit input produces values in range roughly [-1020, 1020]
    # which fits in 12-bit signed range [-2048, 2047]
    gx_norm = np.clip(gx, -2048, 2047).astype(np.int16)
    gy_norm = np.clip(gy, -2048, 2047).astype(np.int16)
    
    return gx_norm, gy_norm

def magnitude_approx_software(gx, gy):
    """
    Software reference magnitude approximation.
    Formula: mag ≈ max(|Gx|, |Gy|) + 0.5 * min(|Gx|, |Gy|)
    Output: 16-bit unsigned (0-65535)
    """
    abs_gx = np.abs(gx.astype(np.int32))
    abs_gy = np.abs(gy.astype(np.int32))
    
    max_v = np.maximum(abs_gx, abs_gy)
    min_v = np.minimum(abs_gx, abs_gy)
    
    # mag = max + (min >> 1) computes in 16-bit + 1 bit = 17-bit
    mag = max_v + (min_v >> 1)
    mag = np.clip(mag, 0, 65535).astype(np.uint16)
    
    return mag

def visualize_magnitude(mag):
    """
    Normalize magnitude to 0-255 for display using actual data range.
    """
    mag_f = mag.astype(np.float32)
    max_val = float(mag_f.max())
    if max_val <= 0.0:
        return np.zeros_like(mag, dtype=np.uint8)
    mag_norm = (mag_f / max_val * 255.0).astype(np.uint8)
    return mag_norm

def send_to_fpga_chunked(port, rgb_img, chunk_rows=None, timeout=60, live_display=False):
    """
    Send RGB pixel data to FPGA in chunks and receive magnitude results.
    FPGA flow: 
    RGB bytes → UART → RGB2Gray → Sobel → Magnitude → UART → Output
    Args:
    port: Serial port (e.g., /dev/ttyUSB1) using Linux
    rgb_img: RGB image array (uint8, shape [H, W, 3], width must be 480 pixels)
    chunk_rows: Number of rows to process per chunk (None = all at once)
    timeout: Read timeout in seconds per chunk
    live_display: Enable real-time visualization of output image
    Returns:
    Magnitude array or nothing on error
    """
    if not SERIAL_AVAILABLE:
        print("ERROR: pyserial not installed. Install with: pip install pyserial")
        return None
    
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=0.1)
        print(f"Connected to {port} at 115200 baud")
        time.sleep(0.1)  # Allow serial to settle
    except serial.SerialException as e:
        print(f"ERROR: Cannot open serial port {port}: {e}")
        return None
    
    height, width, channels = rgb_img.shape
    if channels != 3:
        print(f"ERROR: Expected RGB image with 3 channels, got {channels}")
        ser.close()
        return None
    
    if width != 480:
        print(f"WARNING: Image width is {width}, FPGA expects 480 pixels")
    
    # Determine chunk size
    if chunk_rows is None or chunk_rows >= height:
        chunk_rows = height
        num_chunks = 1
    else:
        num_chunks = (height + chunk_rows - 1) // chunk_rows
    
    print(f"\nProcessing {width}×{height} image in {num_chunks} chunk(s) of {chunk_rows} rows")
    print(f"Using 24-bit RGB transmission (3 bytes/pixel)")
    
    mag_output = np.zeros((height, width), dtype=np.uint8)
    
    # Setup live display if enabled
    display_available = False
    frames_dir = None
    
    if live_display:
        # Check if display is available (X11 or Wayland)
        has_display = bool(os.environ.get('DISPLAY')) or bool(os.environ.get('WAYLAND_DISPLAY'))
        
        if has_display:
            # Try to create OpenCV window
            try:
                cv2.namedWindow('FPGA Output', cv2.WINDOW_NORMAL)
                cv2.imshow('FPGA Output', mag_output)
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
    
    for chunk_idx in range(num_chunks):
        start_row = chunk_idx * chunk_rows
        end_row = min(start_row + chunk_rows, height)
        actual_rows = end_row - start_row
        
        print(f"\n--- Chunk {chunk_idx+1}/{num_chunks}: rows {start_row}-{end_row-1} ({actual_rows} rows) ---")
        
        # Extract chunk
        rgb_chunk = rgb_img[start_row:end_row, :, :]
        
        # Convert RGB to 24-bit byte stream (3 bytes per pixel)
        pixels_rgb = np.zeros((actual_rows, width, 3), dtype=np.uint8)
        pixels_rgb[:, :, 0] = rgb_chunk[:, :, 0]  # R
        pixels_rgb[:, :, 1] = rgb_chunk[:, :, 1]  # G
        pixels_rgb[:, :, 2] = rgb_chunk[:, :, 2]  # B
        
        # Flatten to byte stream
        tx_data = pixels_rgb.flatten()
        num_pixels = actual_rows * width
        tx_bytes = len(tx_data)
        
        print(f"Sending {num_pixels} pixels ({width}×{actual_rows}) = {tx_bytes} bytes")
        
        # Clear any stale data in receive buffer
        ser.reset_input_buffer()
        
        # Send data in larger chunks for better throughput
        CHUNK_SIZE = 4096
        t0 = time.time()
        written = 0
        tx_data_bytes = tx_data.tobytes()
        
        for i in range(0, len(tx_data_bytes), CHUNK_SIZE):
            chunk = tx_data_bytes[i:i+CHUNK_SIZE]
            written += ser.write(chunk)
            ser.flush()
            time.sleep(0.0001)  
        
        tx_dt = time.time() - t0
        
        print(f"Sent {written}/{tx_bytes} bytes in {tx_dt:.3f}s ({written/tx_dt:.0f} B/s)")
        
        # Sobel needs ~480 pixels of line buffer latency = ~2-3 rows at 25MHz
        # Add safety margin: ~2 row latencies
        pipeline_delay = max(0.5, actual_rows * 0.003)  # 3ms per row
        print(f"  Waiting {pipeline_delay:.2f}s for pipeline latency...")
        time.sleep(pipeline_delay)
        
        # Receive magnitude output
        expected_bytes = num_pixels
        print(f"  Receiving magnitude data (expecting {expected_bytes} bytes)...")
        
        rx_buffer = bytearray()
        start_time = time.time()
        last_log = 0
        no_data_count = 0
        max_no_data = 20  # Tolerance for slower responses
        
        # keep reading as long as we get data
        while (time.time() - start_time) < timeout:
            try:
                data = ser.read(min(4096, expected_bytes - len(rx_buffer)))
                if data:
                    rx_buffer.extend(data)
                    no_data_count = 0
                    if len(rx_buffer) - last_log >= 4096:
                        progress_pct = 100 * len(rx_buffer) / expected_bytes
                        print(f"    Received {len(rx_buffer)}/{expected_bytes} bytes ({progress_pct:.1f}%)")
                        last_log = len(rx_buffer)
                    # Stop early if we got all expected data
                    if len(rx_buffer) >= expected_bytes:
                        break
                else:
                    no_data_count += 1
                    # If no data for many consecutive reads, stop waiting
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
        
        # Parse magnitude values
        # FPGA should output one byte per pixel
        mag_chunk = np.frombuffer(rx_buffer, dtype=np.uint8)
        
        # pad or trim to expected size
        if len(mag_chunk) < num_pixels:
            print(f"  WARNING: Incomplete data - expected {num_pixels}, got {len(mag_chunk)}")
            # Pad with black (0)
            mag_chunk = np.pad(mag_chunk, (0, num_pixels - len(mag_chunk)), mode='constant')
        else:
            mag_chunk = mag_chunk[:num_pixels]
        
        mag_chunk = mag_chunk.reshape(actual_rows, width)
        mag_output[start_row:end_row, :] = mag_chunk
        
        # Update live display
        if live_display:
            progress_pct = 100*(chunk_idx+1)/num_chunks
            
            if display_available:
                # Update OpenCV window
                cv2.imshow('FPGA Output', mag_output)
                cv2.waitKey(1)
                print(f"  → Display updated ({progress_pct:.0f}% complete)")
            elif frames_dir:
                # Save frame to directory
                frame_path = frames_dir / f"frame_{chunk_idx+1:04d}.png"
                cv2.imwrite(str(frame_path), mag_output)
                print(f"  → Saved: {frame_path} ({progress_pct:.0f}% complete)")
        
        nonzero = np.count_nonzero(mag_chunk)
        print(f"  Chunk {chunk_idx+1} complete: mag range [{mag_chunk.min()}, {mag_chunk.max()}], {nonzero} non-zero pixels")
    
    ser.close()
    print("\nAll chunks processed!")
    
    # Finalize live display
    if live_display:
        if display_available:
            print("\n✓ Live window updated - Press any key in the window to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif frames_dir:
            print(f"\n✓ Frames saved to: {frames_dir}/")
            print(f"  View with: ls {frames_dir}/ | xargs -I {{}} code {frames_dir}/{{}}")
    
    return mag_output

def main():
    parser = argparse.ArgumentParser(
        description="FPGA Magnitude Image Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python3 fpga_mag_test_image.py --image [inputimage.jpg] --port [portname] --output [outputimage.jpg] --timeout [seconds] --chunk-rows [rows]
        """
    )
    parser.add_argument("--image", required=True, help="Input image path (JPG, PNG, etc.)")
    parser.add_argument("--port", required=True, help="Serial port (e.g., /dev/ttyUSB0, COM3)")
    parser.add_argument("--output", default=None, help="Output magnitude image (default: mag_fpga_<input>.jpg)")
    parser.add_argument("--timeout", type=int, default=60, help="Serial timeout in seconds per chunk (default: 60)")
    parser.add_argument("--width", type=int, default=480, help="Target image width in pixels (default: 480)")
    parser.add_argument("--chunk-rows", type=int, default=None, 
                        help="Process image in chunks of N rows (default: None = all at once). "
                             "Use smaller values (e.g., 50-100) for large images or limited FPGA buffer.")
    parser.add_argument("--no-compare", action='store_true', help="Skip software reference comparison")
    parser.add_argument("--live-display", action='store_true', 
                        help="Display output image in real-time as it's being processed")
    
    args = parser.parse_args()
    
    try:
        # Load image as RGB
        print(f"Loading image: {args.image}")
        rgb_img = load_image(args.image, as_rgb=True)
        orig_h, orig_w, _ = rgb_img.shape
        print(f"Original size: {orig_w}×{orig_h}")
        
        # Resize to requested width (default 480 for FPGA)
        if orig_w != args.width:
            print(f"Resizing to {args.width} pixels wide...")
            rgb_img = resize_to_linewidth(rgb_img, args.width)
        
        height, width, _ = rgb_img.shape
        print(f"Final size: {width}×{height} RGB")
        
        # Software reference (if comparison enabled)
        mag_sw = None
        if not args.no_compare:
            print("\n=== SOFTWARE REFERENCE ===")
            gray = rgb_to_gray_software(rgb_img)
            print(f"Converted to grayscale")
            
            gx, gy = compute_sobel_gradients(gray)
            print(f"Gradient range: Gx [{gx.min()}, {gx.max()}], Gy [{gy.min()}, {gy.max()}]")
            print(f"Gradient types: Gx={gx.dtype}, Gy={gy.dtype}")
            
            mag_sw_16bit = magnitude_approx_software(gx, gy)
            print(f"Magnitude (16-bit) range: {mag_sw_16bit.min()}-{mag_sw_16bit.max()}")
            print(f"Magnitude (16-bit) mean: {mag_sw_16bit.mean():.1f}")
            print(f"Magnitude (16-bit) non-zero: {np.count_nonzero(mag_sw_16bit)}")
            
            # Scale to 8-bit to match FPGA output
            max_mag = max(1, mag_sw_16bit.max())  
            mag_sw = (mag_sw_16bit.astype(np.float32) / max_mag * 255).astype(np.uint8)
            print(f"Magnitude range (8-bit normalized): {mag_sw.min()}-{mag_sw.max()}")
            print(f"Mean magnitude (8-bit): {mag_sw.mean():.1f}")
            print(f"Non-zero (8-bit): {np.count_nonzero(mag_sw)}")
        
        # FPGA test
        print("\n=== FPGA TEST ===")
        mag_fpga = send_to_fpga_chunked(args.port, rgb_img, args.chunk_rows, args.timeout, args.live_display)
        
        if mag_fpga is None:
            print("ERROR: Failed to get FPGA data")
            return 1
        
        print(f"\nFPGA magnitude range: {mag_fpga.min()}-{mag_fpga.max()}")
        print(f"FPGA mean magnitude: {mag_fpga.mean():.1f}")
        
        # Comparison
        if not args.no_compare and mag_sw is not None:
            print("\n=== COMPARISON ===")
            if mag_sw.shape == mag_fpga.shape:
                diff = np.abs(mag_sw.astype(np.int32) - mag_fpga.astype(np.int32))
                print(f"Max difference: {diff.max()}")
                print(f"Mean difference: {diff.mean():.2f}")
                print(f"Std dev difference: {diff.std():.2f}")
                
                match_pct = 100 * np.sum(diff == 0) / diff.size
                within_1_pct = 100 * np.sum(diff <= 1) / diff.size
                within_5_pct = 100 * np.sum(diff <= 5) / diff.size
                within_10_pct = 100 * np.sum(diff <= 10) / diff.size
                
                print(f"Perfect match (diff=0): {match_pct:.1f}%")
                print(f"Within 1: {within_1_pct:.1f}%")
                print(f"Within 5: {within_5_pct:.1f}%")
                print(f"Within 10: {within_10_pct:.1f}%")
                
                # Some variance is expected and ok, I mean it's pretty damn close
                if within_10_pct >= 60:
                    print("✓ PASS: Good agreement between FPGA and software reference")
                elif within_10_pct >= 45:
                    print("⚠ WARN: Moderate agreement - small adjustments may help")
                else:
                    print("✗ FAIL: Poor agreement - verify FPGA design")
            else:
                print(f"ERROR: Shape mismatch - SW: {mag_sw.shape}, FPGA: {mag_fpga.shape}")
        
        # Save outputs
        print("\n=== SAVING OUTPUTS ===")
        
        # output filename
        if args.output:
            output_fpga = args.output
            base = Path(args.output).stem
            output_sw = f"{base}_sw.jpg"
            output_input = f"{base}_input.jpg"
        else:
            base = Path(args.image).stem
            output_sw = f"mag_sw_{base}.jpg"
            output_fpga = f"mag_fpga_{base}.jpg"
            output_input = f"input_{base}.jpg"
        
        # Save input image
        cv2.imwrite(output_input, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        print(f"  Input: {output_input}")
        
        # Save FPGA output
        cv2.imwrite(output_fpga, mag_fpga)
        print(f"  FPGA output: {output_fpga}")
        
        # Save software reference if computed
        if mag_sw is not None:
            cv2.imwrite(output_sw, mag_sw)
            print(f"  Software reference: {output_sw}")
        
        print("\n✓ Processing complete!")
        return 0
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
