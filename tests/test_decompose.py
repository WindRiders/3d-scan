"""测试网格拆解."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.decompose import (
    CutPlane,
    Decomposition,
    brush_select_faces,
    brush_select_radius,
    cut_mesh_multi_plane,
    cut_mesh_with_plane,
    decompose,
    export_parts,
    segment_semantic,
)


@pytest.fixture
def bunny_mesh() -> trimesh.Trimesh:
    """生成类雕像的复合几何体（球+圆柱近似人形）."""
    head = trimesh.creation.icosphere(radius=3.0, subdivisions=2)
    head.apply_translation([0, 0, 12])

    body = trimesh.creation.cylinder(radius=4.0, height=8.0)
    body.apply_translation([0, 0, 4])

    left_arm = trimesh.creation.cylinder(radius=1.0, height=6.0)
    left_arm.apply_translation([-5, 0, 8])

    right_arm = trimesh.creation.cylinder(radius=1.0, height=6.0)
    right_arm.apply_translation([5, 0, 8])

    base = trimesh.creation.cylinder(radius=5.0, height=2.0)
    base.apply_translation([0, 0, -3])

    combined = trimesh.util.concatenate([head, body, left_arm, right_arm, base])
    return combined


def test_segment_semantic_convexity(bunny_mesh: trimesh.Trimesh) -> None:
    """凸性分割产生多个部分."""
    labels = segment_semantic(bunny_mesh, method="convexity", num_parts=5)
    unique = np.unique(labels)
    assert len(unique) >= 2  # 至少两个模块
    assert len(labels) == len(bunny_mesh.faces)


def test_segment_semantic_height(bunny_mesh: trimesh.Trimesh) -> None:
    """高度分层分割."""
    labels = segment_semantic(bunny_mesh, method="height", num_parts=4)
    unique = np.unique(labels)
    assert len(unique) == 4


def test_cut_plane(bunny_mesh: trimesh.Trimesh) -> None:
    """平面切割产生两个有效子网格."""
    plane = CutPlane(
        point=np.array([0.0, 0.0, 6.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    top, bottom = cut_mesh_with_plane(bunny_mesh, plane)
    assert len(top.faces) > 0
    assert len(bottom.faces) > 0


def test_cut_multi_plane(bunny_mesh: trimesh.Trimesh) -> None:
    """多平面切割."""
    planes = [
        CutPlane(np.array([0, 0, 4]), np.array([0, 0, 1])),
        CutPlane(np.array([0, 0, 10]), np.array([0, 0, -1])),
    ]
    pieces = cut_mesh_multi_plane(bunny_mesh, planes)
    assert len(pieces) >= 2


def test_brush_select_faces(bunny_mesh: trimesh.Trimesh) -> None:
    """笔刷选取返回相邻面."""
    # 从第一个面开始生长
    selected = brush_select_faces(bunny_mesh, seed_face=0, max_angle_degrees=30)
    assert len(selected) > 0
    assert len(selected) <= 5000


def test_brush_select_radius(bunny_mesh: trimesh.Trimesh) -> None:
    """球体范围内选取."""
    center = np.array([0.0, 0.0, 12.0])
    selected = brush_select_radius(bunny_mesh, center, radius=5.0)
    assert len(selected) > 0


def test_decompose_full(bunny_mesh: trimesh.Trimesh) -> None:
    """完整拆解流水线."""
    result = decompose(bunny_mesh, method="convexity", num_parts=5)
    assert isinstance(result, Decomposition)
    assert result.part_count >= 2
    assert len(result.face_labels) == len(bunny_mesh.faces)

    # 每个模块至少有一个面
    for part in result.parts:
        assert len(part.face_indices) > 0
        assert part.name


def test_extract_part_mesh(bunny_mesh: trimesh.Trimesh) -> None:
    """提取单个模块网格."""
    result = decompose(bunny_mesh, method="convexity", num_parts=3)
    for part in result.parts:
        sub = result.extract_part_mesh(part.part_id)
        assert len(sub.faces) > 0
        assert len(sub.vertices) > 0


def test_export_parts(bunny_mesh: trimesh.Trimesh, tmp_path: Path) -> None:
    """导出分件 STL."""
    result = decompose(bunny_mesh, method="convexity", num_parts=3)
    paths = export_parts(result, tmp_path / "parts", format="stl")
    assert len(paths) >= 1
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
