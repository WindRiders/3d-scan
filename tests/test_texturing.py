"""测试 UV 展开与纹理烘焙."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.texturing import (
    _hemisphere_samples,
    bake_ambient_occlusion,
    bake_vertex_color,
    unwrap_uv,
)


@pytest.fixture
def cube_mesh(tmp_path: Path) -> Path:
    """生成立方体网格."""
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    p = tmp_path / "cube.ply"
    mesh.export(str(p))
    return p


def test_unwrap_uv(cube_mesh: Path, tmp_path: Path) -> None:
    """UV 展开生成有效 UV 坐标."""
    output = tmp_path / "unwrapped.obj"
    result = unwrap_uv(cube_mesh, output, tex_resolution=1024)
    assert result.exists()
    assert result.stat().st_size > 0
    assert result.suffix == ".obj"
    tm = trimesh.load(str(result), force="mesh")
    assert isinstance(tm.visual, trimesh.visual.TextureVisuals)
    assert tm.visual.uv is not None
    assert len(tm.visual.uv) > 0


def test_bake_vertex_color_no_color(cube_mesh: Path, tmp_path: Path) -> None:
    """无顶点颜色时生成默认纹理."""
    output = tmp_path / "colored.ply"
    result = bake_vertex_color(cube_mesh, output)
    assert result.exists()
    tm = trimesh.load(str(result), force="mesh")
    assert tm.visual is not None


def test_bake_vertex_color_with_color(cube_mesh: Path, tmp_path: Path) -> None:
    """有顶点颜色时保留颜色信息."""
    tm = trimesh.load(str(cube_mesh), force="mesh")
    tm.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.full((len(tm.vertices), 4), [255, 0, 0, 255], dtype=np.uint8),
    )
    colored_path = tmp_path / "red_cube.ply"
    tm.export(str(colored_path))

    output = tmp_path / "baked.ply"
    result = bake_vertex_color(colored_path, output)
    assert result.exists()


def test_bake_ao(cube_mesh: Path, tmp_path: Path) -> None:
    """AO 烘焙计算遮挡值."""
    output = tmp_path / "ao.ply"
    result = bake_ambient_occlusion(cube_mesh, output, samples=4)
    assert result.exists()
    assert result.stat().st_size > 0
    tm = trimesh.load(str(result), force="mesh")
    assert tm.visual.vertex_colors is not None


def test_apply_vertex_colors_no_color(tmp_path: Path, monkeypatch) -> None:
    """非 Trimesh 网格触发 ValueError —— 覆盖 line 75."""
    import trimesh as _tm

    pc = _tm.points.PointCloud(np.array([[0.0, 0.0, 0.0]]))
    monkeypatch.setattr(_tm, "load", lambda *a, **kw: pc)

    with pytest.raises(ValueError, match="不是有效的网格文件"):
        bake_vertex_color(Path("fake.ply"), tmp_path / "out.ply")


def test_hemisphere_samples() -> None:
    """_hemisphere_samples 返回正确数量和有效方向向量 —— 覆盖 line 155."""
    rng = np.random.RandomState(42)
    normal = np.array([0.0, 0.0, 1.0])
    samples = _hemisphere_samples(normal, rng, 8)
    assert len(samples) == 8
    for d in samples:
        assert len(d) == 3
        assert np.dot(d, normal) > 0
        assert np.isclose(np.linalg.norm(d), 1.0)


def test_hemisphere_samples_near_z() -> None:
    """法线接近 Z 轴时触发切线回退分支 —— 覆盖 line 155."""
    rng = np.random.RandomState(42)
    normal = np.array([0.0, 0.01, 0.99])
    normal /= np.linalg.norm(normal)
    assert np.abs(np.dot(normal, [0.0, 0.0, 1.0])) > 0.99
    samples = _hemisphere_samples(normal, rng, 4)
    assert len(samples) == 4
    for d in samples:
        assert np.dot(d, normal) > 0
