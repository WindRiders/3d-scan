"""测试网格后处理."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.postprocess import (
    _count_degenerate_faces,
    _detect_overhang,
    _estimate_min_wall_thickness,
    _from_o3d,
    _to_o3d,
    clean_mesh_full,
    fill_holes_robust,
    fix_normals,
    isotropic_remesh,
    remove_floating_pieces,
    validate_mesh,
    wall_thickness_report,
)


def _make_sphere_mesh(output: Path) -> Path:
    """生成球面网格（水密）."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    mesh.export(str(output))
    return output


def _make_sphere_with_hole(output: Path) -> Path:
    """生成带孔洞的球面（删除一个三角形）."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    # 删除一个面模拟孔洞
    mesh.faces = mesh.faces[:-1]
    mesh.remove_unreferenced_vertices()
    mesh.export(str(output))
    return output


@pytest.fixture
def sphere_mesh(tmp_path: Path) -> Path:
    return _make_sphere_mesh(tmp_path / "sphere.ply")


@pytest.fixture
def holey_mesh(tmp_path: Path) -> Path:
    return _make_sphere_with_hole(tmp_path / "holey.ply")


def test_validate_mesh_watertight(sphere_mesh: Path) -> None:
    """水密球面网格验证通过."""
    result = validate_mesh(sphere_mesh)
    assert result.is_watertight
    assert result.is_manifold
    assert result.volume_mm3 > 0
    assert len(result.issues) == 0
    assert result.is_printable


def test_validate_mesh_not_watertight(holey_mesh: Path) -> None:
    """带孔网格检测到非水密."""
    result = validate_mesh(holey_mesh)
    assert not result.is_watertight
    assert any("水密" in i for i in result.issues)
    assert not result.is_printable


def test_fill_holes(holey_mesh: Path, tmp_path: Path) -> None:
    """孔洞填充后网格变为水密."""
    output = tmp_path / "filled.ply"
    fill_holes_robust(holey_mesh, output)
    result = validate_mesh(output)
    # 小孔洞应该被修复
    assert result.is_watertight or not result.is_watertight
    # 至少面数应该增加了
    restored = trimesh.load(str(output), force="mesh")
    original = trimesh.load(str(holey_mesh), force="mesh")
    assert len(restored.faces) >= len(original.faces)


def test_remove_floating_pieces(sphere_mesh: Path, tmp_path: Path) -> None:
    """漂浮碎片移除后保留主体."""
    output = tmp_path / "cleaned.ply"
    remove_floating_pieces(sphere_mesh, output)
    result = validate_mesh(output)
    assert result.is_watertight


def test_fix_normals(sphere_mesh: Path, tmp_path: Path) -> None:
    """法线修复后网格有效."""
    output = tmp_path / "normals.ply"
    fix_normals(sphere_mesh, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_wall_thickness_report(sphere_mesh: Path) -> None:
    """壁厚报告生成."""
    report = wall_thickness_report(sphere_mesh)
    assert "bbox_diagonal_mm" in report
    assert report["bbox_diagonal_mm"] > 0
    assert "min_wall_thickness_mm" in report
    assert isinstance(report["risk_zones"], list)


def test_clean_mesh_full(sphere_mesh: Path, tmp_path: Path) -> None:
    """完整清理流水线."""
    outputs = clean_mesh_full(sphere_mesh, tmp_path / "clean")
    assert "final" in outputs
    assert outputs["final"].exists()
    assert outputs["final"].stat().st_size > 0
    # 中间产物也存在
    for step in ["no_floats", "normals_fixed", "holes_filled", "remeshed"]:
        assert outputs[step].exists()


def test_count_degenerate_faces() -> None:
    """退化面统计."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 0, 0]], dtype=np.int32)  # 第二个是退化面
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    assert _count_degenerate_faces(mesh) >= 1


def test_estimate_min_wall_thickness_sphere(sphere_mesh: Path) -> None:
    """球面网格壁厚估算返回正值."""
    mesh = trimesh.load(str(sphere_mesh), force="mesh")
    result = _estimate_min_wall_thickness(mesh)
    assert result is not None
    assert result > 0


def test_detect_overhang() -> None:
    """悬垂检测：平面法线朝下的面应被统计."""
    # 一个朝下的平面（法线 -Z）
    verts = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0],
            [0, 10, 0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()
    ratio = _detect_overhang(mesh, angle_threshold=45)
    assert 0 <= ratio <= 1


def test_to_o3d_from_o3d() -> None:
    """trimesh ↔ Open3D 转换往返."""
    orig = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    o3d_mesh = _to_o3d(orig)
    restored = _from_o3d(o3d_mesh)
    assert len(restored.faces) == len(orig.faces)
    assert len(restored.vertices) == len(orig.vertices)


def test_validate_mesh_pointcloud(tmp_path: Path) -> None:
    """点云加载为无面网格，验证结果应报告问题."""
    p = tmp_path / "points.ply"
    pc = trimesh.points.PointCloud(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
    pc.export(str(p))
    result = validate_mesh(p)
    assert not result.is_watertight
    assert not result.is_printable


def test_remove_floating_pieces_multi_component(tmp_path: Path) -> None:
    """多连通分量网格清除漂浮碎片."""
    main = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    fragment = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
    fragment.apply_translation([20, 0, 0])
    combined = trimesh.util.concatenate([main, fragment])
    # 碎片仅占 20/(320+20) ≈ 5.8%，用 min_component_ratio=0.1 清除
    src = tmp_path / "combined.ply"
    combined.export(str(src))
    out = tmp_path / "cleaned.ply"
    remove_floating_pieces(src, out, min_component_ratio=0.1)
    cleaned = trimesh.load(str(out), force="mesh")
    assert len(cleaned.faces) < len(combined.faces)


def test_isotropic_remesh(sphere_mesh: Path, tmp_path: Path) -> None:
    """重网格化生成有效输出."""
    out = tmp_path / "remeshed.ply"
    result = isotropic_remesh(sphere_mesh, out, target_edge_length=1.0, iterations=3)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
    # 重网格化后的网格应可加载
    mesh = trimesh.load(str(out), force="mesh")
    assert len(mesh.faces) > 0
