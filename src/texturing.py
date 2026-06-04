"""UV 展开与纹理烘焙."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

try:
    import xatlas

    _has_xatlas = True
except ImportError:
    xatlas = None  # type: ignore[assignment]
    _has_xatlas = False

logger = logging.getLogger(__name__)


def unwrap_uv(
    mesh_path: Path,
    output_path: Path,
    tex_resolution: int = 2048,
) -> Path:
    """UV 展开：xatlas 优先，否则使用投影回退."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError("不是有效的网格文件")

    if _has_xatlas:
        verts = np.asarray(tm.vertices, dtype=np.float32)
        faces = np.asarray(tm.faces, dtype=np.int32)

        atlas = xatlas.Atlas()
        atlas.add_mesh(verts, faces)
        chart_options = xatlas.ChartOptions()
        pack_options = xatlas.PackOptions()
        pack_options.resolution = tex_resolution
        atlas.generate(chart_options=chart_options, pack_options=pack_options)

        vmapping, indices, uvs = atlas.get_mesh(0)
        logger.info(
            "UV 展开完成 (xatlas): %d 顶点, %d UV 坐标, %d 面",
            len(vmapping),
            len(uvs),
            len(indices),
        )

        out_verts = np.asarray(verts[vmapping])
        out_faces = np.asarray(indices).reshape(-1, 3)
        out_uvs = np.asarray(uvs[:, :2])
        tm_out = trimesh.Trimesh(vertices=out_verts, faces=out_faces)
    else:
        logger.info("xatlas 未安装，使用投影回退 UV")
        tm_out, out_uvs = _projection_unwrap(tm, tex_resolution)

    tm_out.visual = trimesh.visual.TextureVisuals(uv=out_uvs)

    # OBJ 格式保留 UV 坐标, PLY 会丢失
    obj_path = output_path.with_suffix(".obj")
    tm_out.export(str(obj_path))
    logger.info("UV 展开导出: %s", obj_path)
    return obj_path


def _projection_unwrap(
    tm: trimesh.Trimesh,
    tex_resolution: int = 2048,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """投影回退 UV 展开：基于平均法线选择最佳投影平面."""
    normals = np.asarray(tm.vertex_normals, dtype=np.float32)
    avg_normal = normals.mean(axis=0)
    norm = np.linalg.norm(avg_normal)
    if norm > 1e-8:
        avg_normal /= norm
    else:
        avg_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    verts = np.asarray(tm.vertices, dtype=np.float32)

    # 基于法线主导方向选择投影轴
    dominant = int(np.argmax(np.abs(avg_normal)))
    if dominant == 0:
        u, v_values = verts[:, 1], verts[:, 2]
    elif dominant == 1:
        u, v_values = verts[:, 0], verts[:, 2]
    else:
        u, v_values = verts[:, 0], verts[:, 1]

    # 归一化到 [0, 1]
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v_values.min(), v_values.max()
    u_range = u_max - u_min if u_max > u_min else 1.0
    v_range = v_max - v_min if v_max > v_min else 1.0
    uv = np.column_stack([
        (u - u_min) / u_range,
        (v_values - v_min) / v_range,
    ]).astype(np.float32)

    logger.info(
        "UV 展开完成 (投影): %d 顶点, 主导轴=%d, u_range=%.1f, v_range=%.1f",
        len(uv),
        dominant,
        u_range,
        v_range,
    )
    return trimesh.Trimesh(vertices=verts, faces=tm.faces.copy()), uv


def bake_vertex_color(
    mesh_path: Path,
    output_path: Path,
) -> Path:
    """将顶点颜色烘焙为纹理贴图.

    当前用顶点颜色均值模拟（无真实纹理时）。云端部署后用 Blender headless.
    """
    tm = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError("不是有效的网格文件")

    # 检查是否有顶点颜色
    has_color = (
        tm.visual is not None and tm.visual.kind is not None and "vertex" in str(tm.visual.kind)
    )
    if not has_color:
        logger.warning("网格无顶点颜色，生成纯色纹理")
        tm.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.full((len(tm.vertices), 4), [200, 200, 200, 255], dtype=np.uint8),
        )
        tm.export(str(output_path))
        return output_path

    # 有顶点颜色时，烘焙到纹理
    # 简化：转为灰度纹理（生产环境用 Blender Cycles 烘焙）
    colors: np.ndarray | None = None
    visual = tm.visual
    if visual is not None and isinstance(visual, trimesh.visual.ColorVisuals):
        colors = visual.vertex_colors
    if colors is not None:
        gray = np.mean(colors[:, :3], axis=1).astype(np.uint8)
        tm.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.column_stack([gray, gray, gray, np.full(len(gray), 255)]),
        )
    tm.export(str(output_path))
    return output_path


def bake_ambient_occlusion(
    mesh_path: Path,
    output_path: Path,
    samples: int = 256,
) -> Path:
    """用光线投射近似环境光遮蔽 (AO)，写回顶点颜色."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError("不是有效的网格文件")

    # 在顶点处采样 AO
    verts = tm.vertices
    normals = tm.vertex_normals
    ao = np.ones(len(verts), dtype=np.float32)

    # 每个顶点向半球方向投射光线
    for i, (v, n) in enumerate(zip(verts, normals)):
        # 生成半球采样方向
        rng = np.random.RandomState(i)  # 确定性采样
        dirs = _hemisphere_samples(n, rng, samples)
        hits = 0
        for d in dirs:
            loc, _, _ = tm.ray.intersects_location(
                [v + n * 0.001],
                [d],
            )
            if len(loc) > 0:
                dist = np.linalg.norm(loc[0] - v)
                if dist < 10.0:  # 10mm 内视为遮挡
                    hits += 1
        ao[i] = 1.0 - hits / samples

    # 写回顶点颜色
    ao_uint8 = (ao * 255).astype(np.uint8)
    tm.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.column_stack([ao_uint8, ao_uint8, ao_uint8, np.full(len(verts), 255)]),
    )
    tm.export(str(output_path))
    logger.info("AO 烘焙完成: %d 采样/顶点", samples)
    return output_path


def _hemisphere_samples(
    normal: np.ndarray,
    rng: np.random.RandomState,
    n: int,
) -> list[np.ndarray]:
    """生成半球方向采样（法线方向的半球）."""
    # 构建局部坐标系
    up = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(normal, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(normal, up)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    samples = []
    for _ in range(n):
        # Cosine-weighted 半球采样
        u1, u2 = rng.uniform(0, 1, 2)
        r = np.sqrt(u1)
        theta = 2 * np.pi * u2
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sqrt(np.maximum(0, 1 - u1))
        direction = x * tangent + y * bitangent + z * normal
        direction /= np.linalg.norm(direction)
        samples.append(direction)

    return samples
