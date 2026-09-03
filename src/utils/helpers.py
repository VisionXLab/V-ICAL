"""通用辅助函数"""
import base64
import io

import numpy as np
from PIL import Image


def encode_frame_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """将 numpy RGB array 编码为 JPEG bytes"""
    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def frame_to_base64(jpeg_bytes: bytes) -> str:
    """将 JPEG bytes 编码为 base64 字符串"""
    return base64.b64encode(jpeg_bytes).decode("ascii")


def safe_json(v):
    """让 numpy 值可以被 json 序列化（递归处理嵌套结构）"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {safe_json(k): safe_json(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe_json(item) for item in v]
    return v
