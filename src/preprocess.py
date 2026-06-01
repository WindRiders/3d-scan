"""图像预处理：质量检查 + 背景分割."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import ImageConfig

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """图片质量检查结果."""

    file_path: Path
    is_ok: bool
    blur_score: float
    avg_brightness: float
    resolution: tuple[int, int]
    issues: list[str]


def check_blur(image: np.ndarray) -> float:
    """Laplacian 方差法评估模糊度，值越高越清晰."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_brightness(image: np.ndarray) -> float:
    """计算平均亮度 (0-255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def quality_check(file_path: Path, config: ImageConfig) -> QualityReport:
    """对单张图片做质量检查."""
    issues: list[str] = []
    img = cv2.imread(str(file_path))
    if img is None:
        return QualityReport(
            file_path=file_path,
            is_ok=False,
            blur_score=0.0,
            avg_brightness=0.0,
            resolution=(0, 0),
            issues=["无法读取图片"],
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    if w < config.min_resolution[0] or h < config.min_resolution[1]:
        issues.append(f"分辨率过低: {w}x{h}")

    blur = check_blur(img_rgb)
    if blur < config.max_blur_score:
        issues.append(f"模糊度过高: {blur:.1f}")

    brightness = check_brightness(img_rgb)
    if brightness < config.min_brightness:
        issues.append(f"曝光不足: {brightness:.1f}")
    elif brightness > config.max_brightness:
        issues.append(f"过曝: {brightness:.1f}")

    return QualityReport(
        file_path=file_path,
        is_ok=len(issues) == 0,
        blur_score=blur,
        avg_brightness=brightness,
        resolution=(w, h),
        issues=issues,
    )


def quality_check_batch(
    file_paths: list[Path],
    config: ImageConfig,
) -> list[QualityReport]:
    """批量质量检查，返回所有报告."""
    return [quality_check(p, config) for p in file_paths]


def remove_background(
    input_path: Path,
    output_path: Path,
    model_name: str = "u2net",
) -> Path:
    """使用 rembg 去除背景，输出 PNG."""
    try:
        from rembg import new_session, remove
    except ImportError:
        logger.warning("rembg 未安装，跳过背景去除")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(input_path).convert("RGBA")
        img.save(str(output_path), "PNG")
        return output_path

    session = new_session(model_name)
    img = Image.open(input_path).convert("RGBA")
    result = remove(img, session=session)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path), "PNG")
    logger.info("背景已去除: %s → %s", input_path.name, output_path.name)
    return output_path


def preprocess_images(
    file_paths: list[Path],
    work_dir: Path,
    config: ImageConfig,
) -> list[Path]:
    """完整预处理流水线：质检 → 剔除不合格 → 去背景."""
    reports = quality_check_batch(file_paths, config)

    ok_paths = []
    for r in reports:
        if r.is_ok:
            ok_paths.append(r.file_path)
            logger.info("✅ %s (blur=%.1f)", r.file_path.name, r.blur_score)
        else:
            logger.warning("❌ %s: %s", r.file_path.name, "; ".join(r.issues))

    if not ok_paths:
        raise ValueError("所有图片质量不合格，请重新拍摄")

    clean_dir = work_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    clean_paths: list[Path] = []
    for p in ok_paths:
        out = clean_dir / f"{p.stem}_nobg.png"
        remove_background(p, out, config.bg_removal_model)
        clean_paths.append(out)

    return clean_paths
