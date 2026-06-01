"""测试点云 → 网格导出."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from src.config import MeshConfig
from src.mesh_export import (
    estimate_normals,
    load_pointcloud,
    pointcloud_to_mesh,
    poisson_mesh,
    quick_preview_mesh,
    remove_outliers,
    simplify_mesh,
)


@pytest.fixture
def sphere_pcd_path(tmp_path: Path) -> Path:
    """生成球面点云测试文件."""
    n = 10000
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.random.uniform(0, np.pi, n)
    r = 1.0 + 0.02 * np.random.randn(n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    p = tmp_path / "sphere.npy"
    np.save(p, pts)
    return p


@pytest.fixture
def sphere_pcd(sphere_pcd_path: Path) -> o3d.geometry.PointCloud:
    return load_pointcloud(sphere_pcd_path)


def test_load_pointcloud_npy(sphere_pcd_path: Path) -> None:
    """加载 .npy 格式点云."""
    pcd = load_pointcloud(sphere_pcd_path)
    assert len(pcd.points) > 0


def test_remove_outliers(sphere_pcd: o3d.geometry.PointCloud) -> None:
    """统计滤波去离群点."""
    orig = len(sphere_pcd.points)
    cleaned = remove_outliers(sphere_pcd, nb_neighbors=20, std_ratio=2.0)
    # 球面点云几乎无离群点，保留 > 90%
    assert len(cleaned.points) > 0.9 * orig


def test_estimate_normals(sphere_pcd: o3d.geometry.PointCloud) -> None:
    """法线估计."""
    pcd = estimate_normals(sphere_pcd)
    normals = np.asarray(pcd.normals)
    assert normals.shape[1] == 3
    assert not np.any(np.isnan(normals))


def test_poisson_reconstruction(sphere_pcd: o3d.geometry.PointCloud) -> None:
    """Poisson 表面重建."""
    pcd = estimate_normals(sphere_pcd)
    mesh = poisson_mesh(pcd, depth=8)
    assert len(mesh.triangles) > 0
    assert len(mesh.vertices) > 0


def test_simplify_mesh(sphere_pcd: o3d.geometry.PointCloud) -> None:
    """QEM 简化."""
    pcd = estimate_normals(sphere_pcd)
    mesh = poisson_mesh(pcd, depth=8)
    simple = simplify_mesh(mesh, target_faces=500)
    assert len(simple.triangles) <= 500


def test_pointcloud_to_mesh_end_to_end(
    sphere_pcd_path: Path,
    tmp_path: Path,
) -> None:
    """端到端点云→网格."""
    config = MeshConfig(target_faces=2000, voxel_size=0.05)
    ply_path, stl_path = pointcloud_to_mesh(
        sphere_pcd_path,
        tmp_path / "out",
        config,
    )
    assert ply_path.exists()
    assert stl_path.exists()
    assert ply_path.stat().st_size > 0
    assert stl_path.stat().st_size > 0


def test_quick_preview(sphere_pcd_path: Path, tmp_path: Path) -> None:
    """快速预览网格."""
    out = tmp_path / "preview.ply"
    result = quick_preview_mesh(sphere_pcd_path, out, voxel_size=0.05)
    assert result.exists()
    assert result.stat().st_size > 0
