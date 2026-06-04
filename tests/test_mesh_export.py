"""测试点云 → 网格导出."""

from __future__ import annotations

from pathlib import Path

from unittest.mock import patch

import numpy as np
import open3d as o3d
import pytest
import trimesh

from src.config import MeshConfig
from src.mesh_export import (
    estimate_normals,
    fill_holes,
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


def test_load_pointcloud_ply(tmp_path: Path) -> None:
    """加载 .ply 格式点云."""
    pcd = o3d.geometry.PointCloud()
    pts = np.random.randn(100, 3).astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)
    ply_path = tmp_path / "points.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    loaded = load_pointcloud(ply_path)
    assert len(loaded.points) == 100


def test_simplify_mesh_already_small(tmp_path: Path) -> None:
    """简化目标面数大于当前时直接返回原网格."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    result = simplify_mesh(mesh, target_faces=100)
    assert len(result.triangles) == 2


def test_fill_holes_success() -> None:
    """带孔洞的网格填充成功."""
    # 四面体缺底面，留下三角形孔洞
    verts = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 1, 2]], dtype=np.float64
    )
    faces = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    assert not mesh.is_watertight
    result = fill_holes(mesh, max_radius=5.0)
    assert result.is_watertight


def test_load_pointcloud_npy_with_colors(tmp_path: Path) -> None:
    """加载含 rgb 的 .npy (≥6 列)."""
    data = np.random.randn(100, 6).astype(np.float32)
    data[:, :3] *= 10  # 坐标放大
    data[:, 3:6] = np.abs(data[:, 3:6])  # 颜色 ≥ 0
    p = tmp_path / "colored.npy"
    np.save(p, data)
    pcd = load_pointcloud(p)
    assert len(pcd.points) == 100
    assert len(pcd.colors) == 100


def test_fill_holes_no_boundary() -> None:
    """封闭网格无边界时跳过填充."""
    box = trimesh.creation.box(extents=[10, 10, 10])
    with patch("trimesh.repair.fill_holes", side_effect=ValueError("无边界")):
        result = fill_holes(box, max_radius=1.0)
    assert result.is_watertight


def test_voxel_ds_skip_on_too_few(tmp_path: Path) -> None:
    """降采样后点数过少应跳过."""
    # 创建小点云，使降采样后 < 1000 点
    pts = np.random.randn(500, 3).astype(np.float64)
    pcd_path = tmp_path / "small.npy"
    np.save(pcd_path, pts)
    config = MeshConfig(target_faces=100, voxel_size=10.0)
    ply_path, stl_path = pointcloud_to_mesh(pcd_path, tmp_path / "out", config)
    assert ply_path.exists()
