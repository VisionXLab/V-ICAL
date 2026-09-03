"""Video encoding/decoding utilities for AI backends"""

import base64
import io
import tempfile
from pathlib import Path
from typing import List, Optional

from PIL import Image


def frames_to_mp4_base64(frames_b64: List[str], fps: float = 1.0) -> str:
    """
    将 base64 编码的帧序列编码为 MP4 视频，返回 base64 编码的视频

    Args:
        frames_b64: base64 编码的 JPEG 图像列表
        fps: 视频帧率

    Returns:
        base64 编码的 MP4 视频
    """
    import cv2
    import numpy as np

    if not frames_b64:
        raise ValueError("frames_b64 cannot be empty")

    # 将 base64 帧转换为 PIL Image
    pil_images = []
    for b64 in frames_b64:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pil_images.append(img)

    # 创建临时文件
    tmp_path = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)

    try:
        # 获取视频尺寸
        width, height = pil_images[0].size

        # 创建 VideoWriter（使用 H.264 编码以确保浏览器兼容）
        # 优先级：avc1 > H264 > X264 > mp4v
        writer = None
        for codec in ["avc1", "H264", "X264", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
            if test_writer.isOpened():
                writer = test_writer
                break
            else:
                test_writer.release()

        if writer is None:
            raise RuntimeError("No suitable video codec found")

        # 写入所有帧
        for img in pil_images:
            # PIL Image (RGB) -> numpy array (BGR for cv2)
            frame_rgb = np.array(img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

        # 读取视频文件并编码为 base64
        video_bytes = tmp_path.read_bytes()
        return base64.b64encode(video_bytes).decode()

    finally:
        # 清理临时文件
        if tmp_path.exists():
            tmp_path.unlink()


def video_file_to_frames(video_path: Path, fps: Optional[float] = None) -> List[bytes]:
    """
    从视频文件中提取帧，返回 JPEG 编码的帧列表

    Args:
        video_path: 视频文件路径
        fps: 提取帧率（如果为 None，则使用视频原始帧率）

    Returns:
        JPEG 编码的帧列表（bytes）
    """
    import cv2

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    try:
        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps is None:
            fps = video_fps

        # 计算帧间隔
        frame_interval = max(1, int(video_fps / fps)) if fps < video_fps else 1

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 只保留指定间隔的帧
            if frame_idx % frame_interval == 0:
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # numpy array -> PIL Image
                img = Image.fromarray(frame_rgb)
                # PIL Image -> JPEG bytes
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                frames.append(buf.getvalue())

            frame_idx += 1

        return frames

    finally:
        cap.release()
