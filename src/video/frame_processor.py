"""帧和视频处理工具

包含字幕添加、视频生成等功能
"""
import io
import base64
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_op_dir(source: str) -> Path:
    """
    获取操作目录路径

    Args:
        source: "ai" / "human" / "crossval"

    Returns:
        对应的操作目录路径
    """
    base_dir = Path(__file__).parent.parent.parent
    if source == "ai":
        return base_dir / "ai_operation"
    if source == "crossval":
        return base_dir / "crossval_operation"
    return base_dir / "human_operation"


def get_caption_font(size: int = 16):
    """获取字幕字体，优先 CJK 兼容字体，fallback 到默认"""
    candidates = [
        # Windows — CJK 字体优先
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",      # 微软雅黑粗体
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def add_caption(
    img: Image.Image,
    caption: str,
    bar_h: int = 40,
    font_size: int = 16,
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """在图片底部添加黑色字幕条和指定颜色的居中文字。"""
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
    canvas.paste(img, (0, 0))

    font = get_caption_font(font_size)
    draw = ImageDraw.Draw(canvas)

    # 自动换行
    pad_x = 8
    max_w = w - 2 * pad_x
    words = caption.split()
    lines, cur = [], ""
    for word in words:
        test = (cur + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [""]

    # 垂直居中
    line_heights = []
    total_th = 0
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font)
        lh = bb[3] - bb[1]
        line_heights.append(lh)
        total_th += lh
    total_th += (len(lines) - 1) * 4

    y = h + (bar_h - total_th) // 2
    for ln, lh in zip(lines, line_heights):
        bb = draw.textbbox((0, 0), ln, font=font)
        tw = bb[2] - bb[0]
        x = (w - tw) // 2
        # 黑描边 + 指定颜色文字
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0))
        draw.text((x, y), ln, font=font, fill=text_color)
        y += lh + 4

    return canvas


def save_video(pil_images: list, output_path: Path, fps: float) -> str:
    """将 PIL Image 列表保存为 mp4 视频，返回文件名或错误信息"""
    if not pil_images:
        return "no frames"
    try:
        import cv2
    except ImportError:
        return "cv2 not installed, video skipped"
    try:
        first = pil_images[0]
        w, h = first.size
        # 尝试使用 H.264 编码（浏览器兼容性最好）
        # 优先级：avc1 > H264 > X264 > mp4v
        writer = None
        for codec in ["avc1", "H264", "X264", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if test_writer.isOpened():
                print(f"[Video] Using codec: {codec}")
                writer = test_writer
                break
            else:
                test_writer.release()

        if writer is None:
            return "no suitable video codec found"

        for img in pil_images:
            frame = np.array(img)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return output_path.name
    except Exception as e:
        return f"video error: {e}"


ENDING_TYPE_CONFIG = {
    "demo_end":  {"label": "Demo End",  "color": (100, 149, 237)},
    "victory":   {"label": "Victory",   "color": (50, 200, 50)},
    "game_over": {"label": "Game Over", "color": (220, 50, 50)},
    "gameover":  {"label": "Game Over", "color": (220, 50, 50)},
    "time_out":  {"label": "Time Out",  "color": (255, 165, 0)},
    "timeout":   {"label": "Time Out",  "color": (255, 165, 0)},
    "draw": {"label":"Draw","color":(255,255,255)}
}


def add_ending_label(img: Image.Image, ending_type: str) -> Image.Image:
    """在画面上叠加结束类型标注（半透明黑条 + 彩色大字）"""
    config = ENDING_TYPE_CONFIG.get(ending_type)
    if not config:
        return img

    label = config["label"]
    color = config["color"]
    w, h = img.size

    # RGBA overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 半透明黑条：画面 35% 高度处，高度为图片高度的 20%
    bar_y = int(h * 0.35)
    bar_h = max(30, int(h * 0.20))
    draw.rectangle([(0, bar_y), (w, bar_y + bar_h)], fill=(0, 0, 0, 140))

    # 字体大小自适应
    font_size = max(16, min(80, w // 6))
    font = get_caption_font(font_size)

    # 文字居中
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = bar_y + (bar_h - th) // 2

    # 黑色描边
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((tx + dx, ty + dy), label, font=font, fill=(0, 0, 0, 220))

    # 彩色文字
    draw.text((tx, ty), label, font=font, fill=color + (255,))

    # 合成
    base = img.convert("RGBA")
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def load_frames_from_source(op_dir: Path, folder_name: str, frame_source: str) -> List[str]:
    """
    从指定的帧源加载帧（original/subbed/subbed_nl）
    如果指定的源不存在，则回退到 frames/ 目录

    Args:
        op_dir: 操作目录（human_operation 或 ai_operation）
        folder_name: 文件夹名称
        frame_source: 帧源类型（original/subbed/subbed_nl）

    Returns:
        base64 编码的帧列表
    """
    source_map = {
        "original": "frames",
        "subbed": "subbed",
        "subbed_nl": "subbed_nl"
    }
    subfolder = source_map.get(frame_source, "frames")
    frames_dir = op_dir / folder_name / subfolder

    # 如果指定的文件夹不存在，回退到 frames/
    if not frames_dir.exists():
        frames_dir = op_dir / folder_name / "frames"

    frames_b64 = []
    for img_path in sorted(frames_dir.glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        frames_b64.append(b64)

    return frames_b64
