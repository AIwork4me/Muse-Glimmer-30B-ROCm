#!/usr/bin/env python3
"""Generate a fixed 128x128 RGB PNG test image for Study 3 (vision), stdlib-only."""
import struct
import zlib

W = H = 128


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def main(path):
    raw = bytearray()
    for y in range(H):
        raw.append(0)  # filter type 0 (None) per scanline
        for x in range(W):
            r = (x * 2) & 255
            g = (y * 2) & 255
            b = ((x + y)) & 255
            raw += bytes((r, g, b))
    png = (b"\x89PNG\r\n\x1a\n" +
           _chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
           _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) +
           _chunk(b"IEND", b""))
    open(path, "wb").write(png)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "test-image.png")
