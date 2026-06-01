"""测试连接件生成."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.connectors import (
    _extrude_polygon,
    _rotation_from_to,
    export_connectors,
    generate_dovetail_pair,
    generate_magnet_slot,
    generate_pin_hole_pair,
    generate_snap_fit,
)


def test_generate_dovetail() -> None:
    """燕尾榫生成有效 mesh."""
    tail, socket = generate_dovetail_pair(
        joint_center=np.array([0.0, 0.0, 0.0]),
        joint_normal=np.array([0.0, 0.0, 1.0]),
        tail_width=5.0,
        tail_depth=8.0,
    )
    assert len(tail.faces) > 0
    assert len(socket.faces) > 0
    assert tail.is_watertight
    assert socket.is_watertight
    assert tail.volume > 0
    assert socket.volume > 0


def test_generate_pin_hole() -> None:
    """圆柱销+孔生成."""
    pin, hole = generate_pin_hole_pair(
        position=np.array([5.0, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        pin_diameter=4.0,
        pin_length=10.0,
    )
    assert len(pin.faces) > 0
    assert len(hole.faces) > 0
    assert pin.is_watertight


def test_generate_magnet_slot() -> None:
    """磁铁槽生成."""
    result = generate_magnet_slot(
        position=np.array([0.0, 0.0, 5.0]),
        face_normal=np.array([0.0, 0.0, 1.0]),
        diameter=6.1,
        depth=3.0,
    )
    assert len(result.faces) > 0
    assert result.is_watertight


def test_generate_snap_fit() -> None:
    """卡扣生成."""
    result = generate_snap_fit(
        base_position=np.array([0.0, 0.0, 5.0]),
        engagement_direction=np.array([0.0, 1.0, 0.0]),
        beam_width=4.0,
        beam_length=8.0,
        beam_thickness=1.5,
        hook_height=2.0,
    )
    assert len(result.faces) > 0


# ── 辅助函数测试 ──────────────────────────────────────────────


def test_extrude_polygon_circle() -> None:
    """圆形多边形挤出生成水密 mesh."""
    theta = np.linspace(0, 2 * np.pi, 33)[:-1]  # 不含重复终点
    poly2d = np.column_stack([np.cos(theta), np.sin(theta)]) * 3.0
    poly = np.column_stack([poly2d, np.zeros(len(poly2d))])
    mesh = _extrude_polygon(poly, direction=np.array([0.0, 0.0, 1.0]), height=5.0)
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_extrude_polygon_square() -> None:
    """矩形多边形挤出."""
    poly2d = np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float)
    poly = np.column_stack([poly2d, np.zeros(len(poly2d))])
    mesh = _extrude_polygon(poly, direction=np.array([0.0, 0.0, 1.0]), height=3.0)
    assert mesh.is_watertight
    # 体积 ≈ 4 * 2 * 3 = 24
    assert 20 < mesh.volume < 30


def test_rotation_from_to_identity() -> None:
    """同方向旋转矩阵为单位阵."""
    v = np.array([0.0, 0.0, 1.0])
    R = _rotation_from_to(v, v)
    np.testing.assert_allclose(R[:3, :3], np.eye(3), atol=1e-6)


def test_rotation_from_to_90deg() -> None:
    """90 度旋转: Z → X."""
    R = _rotation_from_to(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    result = R[:3, :3] @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)


def test_rotation_from_to_opposite() -> None:
    """反向旋转: Z → -Z."""
    R = _rotation_from_to(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
    result = R[:3, :3] @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-6)


# ── 集成测试 ──────────────────────────────────────────────────


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    """测试用立方体."""
    return trimesh.creation.box(extents=[50, 50, 50])


def test_export_connectors_creates_stl(box_mesh: trimesh.Trimesh, tmp_path: Path) -> None:
    """导出连接件生成 STL 文件."""
    output_dir = tmp_path / "connectors"
    results = export_connectors([box_mesh], output_dir)
    assert len(results) == 1
    assert results["part_0"].exists()
    assert results["part_0"].suffix == ".stl"


def test_dovetail_placement_on_box(box_mesh: trimesh.Trimesh) -> None:
    """燕尾榫生成在立方体面上."""
    tail, socket = generate_dovetail_pair(
        joint_center=np.array([25.0, 0.0, 0.0]),  # 在 X+ 面上
        joint_normal=np.array([1.0, 0.0, 0.0]),
        tail_width=5.0,
        tail_depth=8.0,
    )
    # tail 应在 X+ 方向突出
    tail_centroid = np.mean(tail.vertices, axis=0)
    assert tail_centroid[0] > 25.0  # 突出于盒面


def test_pin_placement_on_box(box_mesh: trimesh.Trimesh) -> None:
    """销+孔生成在立方体面上."""
    pin, hole = generate_pin_hole_pair(
        position=np.array([0.0, 25.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        pin_diameter=4.0,
        pin_length=10.0,
    )
    assert pin.is_watertight
    # hole 应有内表面（非完全封闭的圆柱）
    assert hole.volume > 0
