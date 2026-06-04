"""测试图片预处理."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

from src.config import ImageConfig
from src.preprocess import (
    check_blur,
    check_brightness,
    preprocess_images,
    quality_check,
    quality_check_batch,
    remove_background,
)


def _make_image(width: int = 400, height: int = 300) -> Path:
    """创建带纹理的测试图片（模拟真实照片）."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    y, x = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    mask = ((x - cx) ** 2 + (y - cy) ** 2) < (min(width, height) // 3) ** 2
    arr[mask] = [100, 150, 200]
    img = Image.fromarray(arr)
    p = Path("/tmp/test_img.jpg")
    img.save(p)
    return p


def _make_blurry_image(width: int = 400, height: int = 300) -> Path:
    """创建模糊图片 — 高斯模糊使 Laplacian 方差极低."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(arr, (51, 51), 20)
    p = Path("/tmp/test_blurry.jpg")
    cv2.imwrite(str(p), blurred)
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


def test_quality_check_blur_fail() -> None:
    """模糊度过高应被拒绝."""
    p = _make_blurry_image(800, 600)
    cfg = ImageConfig(min_resolution=(512, 512))
    report = quality_check(p, cfg)
    assert not report.is_ok
    assert any("模糊" in i for i in report.issues)


def test_quality_check_underexposed() -> None:
    """曝光不足应被检测."""
    img = np.full((600, 800, 3), 5, dtype=np.uint8)
    p = Path("/tmp/test_dark.jpg")
    Image.fromarray(img).save(p)
    cfg = ImageConfig(min_resolution=(512, 512), min_brightness=20.0)
    report = quality_check(p, cfg)
    assert not report.is_ok
    assert any("曝光不足" in i for i in report.issues)


def test_quality_check_overexposed() -> None:
    """过曝应被检测."""
    img = np.full((600, 800, 3), 250, dtype=np.uint8)
    p = Path("/tmp/test_bright.jpg")
    Image.fromarray(img).save(p)
    cfg = ImageConfig(min_resolution=(512, 512), max_brightness=240.0)
    report = quality_check(p, cfg)
    assert not report.is_ok
    assert any("过曝" in i for i in report.issues)


def test_remove_background_basic() -> None:
    """背景去除基本流程."""
    p = _make_image(800, 600)
    out = Path("/tmp/test_nobg.png")
    result = remove_background(p, out)
    assert result == out
    assert out.exists()
    assert out.suffix == ".png"


def test_preprocess_images_basic(tmp_path: Path) -> None:
    """完整预处理流水线."""
    # 创建合格的测试图片
    imgs = []
    for i in range(2):
        arr = np.random.randint(50, 200, (600, 800, 3), dtype=np.uint8)
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(arr).save(p)
        imgs.append(p)
    cfg = ImageConfig(min_resolution=(512, 512))
    results = preprocess_images(imgs, tmp_path / "work", cfg)
    assert len(results) == 2
    assert all(r.exists() for r in results)
    assert all(r.suffix == ".png" for r in results)


def test_preprocess_images_all_rejected(tmp_path: Path) -> None:
    """全部不合格应抛异常."""
    # 小图（分辨率过低）
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    p = tmp_path / "tiny.jpg"
    Image.fromarray(arr).save(p)
    cfg = ImageConfig(min_resolution=(512, 512))
    try:
        preprocess_images([p], tmp_path / "work", cfg)
        assert False, "应抛出异常"
    except ValueError as e:
        assert "质量不合格" in str(e)


def test_remove_background_with_named_model(tmp_path: Path) -> None:
    """指定 model_name 时调用 new_session 路径."""
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    Image.fromarray(arr).save(img_path)
    out_path = tmp_path / "test_nobg.png"

    fake_rembg = MagicMock()
    fake_rembg.new_session.return_value = MagicMock()
    fake_rembg.remove.return_value = Image.fromarray(arr).convert("RGBA")

    with patch.dict("sys.modules", {"rembg": fake_rembg}):
        result = remove_background(img_path, out_path, model_name="u2net")
        fake_rembg.new_session.assert_called_once_with("u2net")
        fake_rembg.remove.assert_called_once()
        assert result == out_path
        assert out_path.exists()
