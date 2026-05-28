"""测试工具函数."""

from __future__ import annotations

import numpy as np

from src.utils import (
    estimate_point_count,
    image_hash,
    normalize_points,
    task_id,
)


def test_image_hash() -> None:
    """图片哈希."""
    h1 = image_hash(b"test data")
    h2 = image_hash(b"test data")
    h3 = image_hash(b"different")
    assert len(h1) == 16
    assert h1 == h2
    assert h1 != h3


def test_task_id() -> None:
    """任务 ID 生成."""
    ids = {task_id() for _ in range(100)}
    assert len(ids) == 100  # 全部唯一


def test_normalize_points() -> None:
    """点云归一化."""
    pts = np.array([[10.0, 0, 0], [-10.0, 0, 0], [0, 10.0, 0]], dtype=np.float64)
    normed = normalize_points(pts)
    # 中心应在原点
    assert np.allclose(normed.mean(axis=0), 0, atol=1e-7)
    # 最远点距离为 1
    max_dist = np.max(np.linalg.norm(normed, axis=1))
    assert np.isclose(max_dist, 1.0)


def test_estimate_point_count() -> None:
    """估算采样点数."""
    count = estimate_point_count(density_mm=1.0, bbox_diag_mm=100)
    assert 400_000 < count < 600_000  # 约 523k
