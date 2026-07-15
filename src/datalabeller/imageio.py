"""ROS image message decoding (no cv_bridge) and perceptual hashing."""
from __future__ import annotations

import numpy as np
import cv2

# Common Bayer encodings -> cv2 debayer code (approximate; refine per camera).
_BAYER = {
    "bayer_rggb8": cv2.COLOR_BAYER_BG2BGR,
    "bayer_bggr8": cv2.COLOR_BAYER_RG2BGR,
    "bayer_gbrg8": cv2.COLOR_BAYER_GR2BGR,
    "bayer_grbg8": cv2.COLOR_BAYER_GB2BGR,
}


def image_msg_to_bgr(msg) -> np.ndarray:
    """Decode a sensor_msgs/Image message into an HxWx3 BGR uint8 array."""
    enc = msg.encoding.lower()
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        ch = 4 if enc.endswith("a8") else 3
        arr = buf.reshape(h, step // 1)[:, : w * ch].reshape(h, w, ch)
        if enc.startswith("rgb"):
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if ch == 4 else cv2.COLOR_RGB2BGR)
        if ch == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(arr)

    if enc in ("mono8", "8uc1"):
        arr = buf.reshape(h, step)[:, :w]
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if enc in ("mono16", "16uc1"):
        arr = buf.view(np.uint16).reshape(h, step // 2)[:, :w]
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if enc in _BAYER:
        arr = buf.reshape(h, step)[:, :w]
        return cv2.cvtColor(arr, _BAYER[enc])

    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


def compressed_msg_to_bgr(msg) -> np.ndarray:
    """Decode a sensor_msgs/CompressedImage (jpeg/png) into HxWx3 BGR."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode CompressedImage (format={msg.format!r})")
    return img


def ahash(bgr: np.ndarray, size: int = 8) -> int:
    """64-bit average hash. Cheap, robust enough to drop near-duplicate frames."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
