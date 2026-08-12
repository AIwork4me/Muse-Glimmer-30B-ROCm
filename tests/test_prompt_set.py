import json
import os
import struct
import zlib

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROMPT_JSON = os.path.join(ROOT, "scripts", "prompt-sets", "muse-glimmer-diverse.json")
IMG = os.path.join(ROOT, "scripts", "prompt-sets", "test-image.png")

CATEGORIES = {"code", "math", "factual", "creative", "reasoning", "instruction"}


def test_prompt_set_schema():
    d = json.load(open(PROMPT_JSON))
    assert d["id"] == "muse-glimmer-diverse"
    assert d["version"] == 1
    cats = {p["category"] for p in d["prompts"]}
    assert cats == CATEGORIES
    for p in d["prompts"]:
        assert isinstance(p["text"], str) and len(p["text"]) > 20


def test_test_image_is_valid_png():
    raw = open(IMG, "rb").read()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR is the first chunk; width/height are big-endian uint32 at bytes 16..24
    assert raw[12:16] == b"IHDR"
    w, h = struct.unpack(">II", raw[16:24])
    assert w >= 64 and h >= 64
    # Validate we can find IDAT chunk and decompress it (basic PNG integrity check)
    pos = 8
    found_idat = False
    while pos < len(raw):
        chunk_len = struct.unpack(">I", raw[pos:pos+4])[0]
        chunk_type = raw[pos+4:pos+8]
        if chunk_type == b"IDAT":
            # Try to decompress the IDAT data to ensure PNG isn't truncated
            idat_data = raw[pos+8:pos+8+chunk_len]
            zlib.decompress(idat_data)
            found_idat = True
            break
        pos += 12 + chunk_len
    assert found_idat, "IDAT chunk not found in PNG"
