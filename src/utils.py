"""通用工具函数."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np


def image_hash(data: bytes) -> str:
    """对图片字节计算 SHA256 摘要（前 16 位）."""
    return hashlib.sha256(data).hexdigest()[:16]


def task_id() -> str:
    """生成唯一任务 ID."""
    return uuid.uuid4().hex[:12]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_points(points: np.ndarray) -> np.ndarray:
    """将点云归一化到单位球内."""
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-8:
        points /= scale
    return points


def estimate_point_count(density_mm: float, bbox_diag_mm: float) -> int:
    """根据密度（mm）和包围盒对角线估算均匀采样所需点数."""
    # 球面体量近似：4/3 π (r/density)^3
    r = bbox_diag_mm / 2
    return int((4 / 3) * np.pi * (r / density_mm) ** 3)
