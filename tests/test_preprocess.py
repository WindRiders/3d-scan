"""测试图片预处理."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import ImageConfig
from src.preprocess import (
    check_blur,
    check_brightness,
    quality_check,
    quality_check_batch,
)


def _make_image(width: int = 400, height: int = 300) -> Path:
    """创建带纹理的测试图片（模拟真实照片）."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # 添加渐变模拟"物体"
    y, x = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    mask = ((x - cx) ** 2 + (y - cy) ** 2) < (min(width, height) // 3) ** 2
    arr[mask] = [100, 150, 200]
    img = Image.fromarray(arr)
    p = Path("/tmp/test_img.jpg")
    img.save(p)
    return p


def test_check_blur_clear_image() -> None:
    """清晰图片模糊度低（方差大）."""
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    score = check_blur(img)
    # 噪声图片的 Laplacian 方差应该较高
    assert score > 0


def test_check_blur_uniform_image() -> None:
    """纯色图片模糊度高（方差小）."""
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    score = check_blur(img)
    assert score < 1.0


def test_check_brightness_normal() -> None:
    """正常亮度."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    b = check_brightness(img)
    assert 127 < b < 129


def test_quality_check_pass() -> None:
    """质检通过."""
    p = _make_image(800, 600)
    cfg = ImageConfig(min_resolution=(512, 512))
    report = quality_check(p, cfg)
    assert report.is_ok
    assert len(report.issues) == 0


def test_quality_check_low_resolution() -> None:
    """质检拒绝低分辨率."""
    p = _make_image(200, 200)
    cfg = ImageConfig(min_resolution=(512, 512))
    report = quality_check(p, cfg)
    assert not report.is_ok
    assert any("分辨率" in i for i in report.issues)


def test_quality_check_batch() -> None:
    """批量质检."""
    paths = [_make_image(800, 600), _make_image(800, 600)]
    cfg = ImageConfig()
    reports = quality_check_batch(paths, cfg)
    assert len(reports) == 2
    assert all(r.is_ok for r in reports)


def test_quality_check_unreadable() -> None:
    """无法读取的文件."""
    report = quality_check(Path("/nonexistent/img.jpg"), ImageConfig())
    assert not report.is_ok
    assert any("无法读取" in i for i in report.issues)
