#!/usr/bin/env python3
"""Generate a simple favicon for the frontend."""

import struct
import os

def create_simple_favicon():
    """Create a simple 16x16 blue favicon.ico"""
    # Create a minimal ICO file (16x16, 32-bit ARGB)

    # ICO header
    ico = bytearray()
    ico.extend(struct.pack('<HHH', 0, 1, 1))  # Reserved, Type (1=ICO), Count

    # Icon directory entry
    ico.extend(struct.pack('<BBBBHHII',
        16,      # width
        16,      # height
        0,       # color count (0 = no palette)
        0,       # reserved
        1,       # color planes
        32,      # bits per pixel
        1024,    # size of image data
        22       # offset to image data
    ))

    # Create a simple 16x16 blue square image (very minimal)
    # BMP header for the image
    image = bytearray()
    image.extend(struct.pack('<I', 40))  # BMP header size
    image.extend(struct.pack('<i', 16))  # width
    image.extend(struct.pack('<i', 32))  # height (doubled for BMP)
    image.extend(struct.pack('<HH', 1, 32))  # planes, bits per pixel
    image.extend(struct.pack('<I', 0))   # compression
    image.extend(struct.pack('<I', 0))   # image size
    image.extend(struct.pack('<ii', 0, 0))  # x/y pixels per meter
    image.extend(struct.pack('<II', 0, 0))  # colors used/important

    # Add 16x16 blue pixels (32 bits each - BGRA format)
    # Blue color: 0xFF0000FF (BGRA)
    for y in range(32):  # 32 rows (16 pixels + 16 for alpha mask)
        for x in range(16):
            if y < 16:
                image.extend(b'\xFF\x00\x00\xFF')  # Blue pixel (BGRA)
            else:
                image.extend(b'\x00\x00\x00\x00')  # Transparent (for mask)

    ico.extend(image)

    # Write to file
    output_path = os.path.join(os.path.dirname(__file__), 'favicon.ico')
    with open(output_path, 'wb') as f:
        f.write(ico)

    print(f"✓ favicon.ico created successfully at: {output_path}")
    print(f"  Size: {len(ico)} bytes")

if __name__ == '__main__':
    create_simple_favicon()
