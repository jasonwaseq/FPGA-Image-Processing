#!/usr/bin/env python3
"""
Software Magnitude Test Reference Implementation
Usage:
  python3 fpga_mag_test.py [--image car.jpg] [--output mag_output.jpg]
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

def load_image(image_path):
    """Load image and convert to grayscale."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def compute_sobel_gradients(gray):
    """Compute Sobel gradients matching FPGA 12-bit signed output."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Scale to 12-bit signed range used by sobel module
    gx_norm = np.clip(gx / 4, -2048, 2047).astype(np.int16)
    gy_norm = np.clip(gy / 4, -2048, 2047).astype(np.int16)
    
    return gx_norm, gy_norm

def magnitude_approx(gx, gy):
    """
    Software implementation of FPGA magnitude approximation.
    mag ≈ max(|Gx|, |Gy|) + 0.5 * min(|Gx|, |Gy|)
    Output: 16-bit unsigned (0-65535)
    """
    abs_gx = np.abs(gx.astype(np.int32))
    abs_gy = np.abs(gy.astype(np.int32))
    
    max_v = np.maximum(abs_gx, abs_gy)
    min_v = np.minimum(abs_gx, abs_gy)
    
    # Magnitude: max + min/2
    mag = max_v + (min_v >> 1)
    
    # Saturate to 16-bit unsigned
    mag = np.clip(mag, 0, 65535).astype(np.uint16)
    
    return mag

def visualize_magnitude(mag):
    """
    Normalize magnitude to 0-255 for display using the actual data range.
    Input: 16-bit unsigned magnitude
    Output: 8-bit unsigned image
    """
    mag_f = mag.astype(np.float32)
    maxv = float(mag_f.max())
    if maxv <= 0.0:
        return np.zeros_like(mag, dtype=np.uint8)
    mag_norm = (mag_f / maxv * 255.0).astype(np.uint8)
    return mag_norm

def main():
    parser = argparse.ArgumentParser(description="FPGA Magnitude Test - Software Reference")
    parser.add_argument("--image", default="../../jupyter/car.jpg",
                        help="Path to input image (default: ../../jupyter/car.jpg)")
    parser.add_argument("--output", default="mag_output.jpg",
                        help="Output magnitude image filename")
    parser.add_argument("--scale", type=int, default=1,
                        help="Downsample factor (default: 1)")
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image: {args.image}")
    gray = load_image(args.image)
    height, width = gray.shape
    print(f"Image size: {width}x{height}")
    
    # Compute Sobel gradients
    print("Computing Sobel gradients...")
    gx, gy = compute_sobel_gradients(gray)
    print(f"Gradient range: Gx [{gx.min()}, {gx.max()}], Gy [{gy.min()}, {gy.max()}]")
    
    # Downsample if requested
    if args.scale > 1:
        print(f"Downsampling by {args.scale}x...")
        gx = gx[::args.scale, ::args.scale]
        gy = gy[::args.scale, ::args.scale]
        print(f"Downsampled to: {gx.shape}")
    
    # Compute magnitude
    print("Computing magnitude approximation...")
    mag = magnitude_approx(gx, gy)
    
    # Visualize and save
    print("Creating visualization...")
    mag_vis = visualize_magnitude(mag)
    cv2.imwrite(args.output, mag_vis)
    
    # Output statistics
    print("\n=== Magnitude Statistics ===")
    print(f"Range: {mag.min()}-{mag.max()} (16-bit: 0-65535)")
    print(f"Mean:  {mag.mean():.1f}")
    print(f"Std:   {mag.std():.1f}")
    print(f"Output saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
