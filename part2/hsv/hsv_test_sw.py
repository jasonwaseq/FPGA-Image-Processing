#!/usr/bin/env python3
"""
Software HSV Image Processing

Loads an image, resizes to 480 pixels wide, computes grayscale, Sobel gradients, magnitude,
then saves the HSV visualization as an output image.

Usage:
    python3 hsv_test_sw.py --image [inputimage.jpg] --output [outputimage.jpg]
"""
import cv2
import numpy as np
import argparse
from pathlib import Path

def load_image(image_path, as_rgb=True):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    if as_rgb:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_to_width(img, width=480):
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h))

def compute_sobel_gradients(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    gx_norm = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gy_norm = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    mag_norm = np.clip(mag, 0, 255).astype(np.uint8)
    return gx_norm, gy_norm, mag_norm

def arctan2_approx(gy, gx):
    abs_gx = np.abs(gx.astype(np.float32))
    abs_gy = np.abs(gy.astype(np.float32))
    denom = abs_gx + abs_gy
    denom[denom == 0] = 1
    base_angle = (abs_gy * 90.0) / denom
    angle = np.where(gx >= 0, base_angle, 180.0 - base_angle)
    return angle.astype(np.uint8)

def sigmoid_lut(mag_values):
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
    h = arctan2_approx(gy, gx)
    v = sigmoid_lut(mag)
    s = np.full_like(h, 255)
    return h, s, v

def create_hsv_visualization(h, s, v):
    hsv_img = np.dstack((h, s, v)).astype(np.uint8)
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return rgb_img

def main():
    parser = argparse.ArgumentParser(description="Software HSV Test - Process Image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Output filename (default: hsv_sw_<input>.jpg)")
    parser.add_argument("--width", type=int, default=480, help="Resize to width (default: 480)")
    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    rgb_img = load_image(args.image, as_rgb=True)
    height, width = rgb_img.shape[:2]
    print(f"Original size: {width}×{height}")
    rgb_img = resize_to_width(rgb_img, args.width)
    height, width = rgb_img.shape[:2]
    print(f"Resized to: {width}×{height} RGB")

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

    if args.output:
        output_sw = args.output
    else:
        base = Path(args.image).stem
        output_sw = f"hsv_sw_{base}.jpg"
    cv2.imwrite(output_sw, cv2.cvtColor(hsv_sw, cv2.COLOR_RGB2BGR))
    print(f"Output saved: {output_sw}")

if __name__ == "__main__":
    main()
