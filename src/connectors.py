"""连接件生成：燕尾榫、圆柱销、磁铁槽、卡扣."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class ConnectorSpec:
    """连接件规格."""
    type: str  # dovetail / pin_hole / magnet / snap_fit
    position: np.ndarray
    orientation: np.ndarray
    dimensions: dict


# ── 燕尾榫 (Dovetail) ──

def generate_dovetail_pair(
    joint_center: np.ndarray,
    joint_normal: np.ndarray,
    joint_up: np.ndarray | None = None,
    tail_width: float = 5.0,
    tail_depth: float = 8.0,
    tail_angle_degrees: float = 15.0,
    extrusion_height: float = 4.0,
    clearance: float = 0.15,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """生成燕尾榫头和榫槽（独立 mesh，不修改原网格）."""
    joint_normal = joint_normal / np.linalg.norm(joint_normal)

    if joint_up is None:
        joint_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(joint_up, joint_normal)) > 0.9:
            joint_up = np.array([1.0, 0.0, 0.0])

    tangent = np.cross(joint_normal, joint_up)
    tangent /= np.linalg.norm(tangent)
    joint_up = np.cross(tangent, joint_normal)
    joint_up /= np.linalg.norm(joint_up)

    angle_rad = np.radians(tail_angle_degrees)
    half_w = tail_width / 2
    inset = clearance / 2

    # 榫头梯形顶点
    local_verts = np.array([
        [-half_w + inset, 0, 0],
        [-half_w + inset + tail_depth * np.tan(angle_rad), tail_depth, 0],
        [half_w - inset - tail_depth * np.tan(angle_rad), tail_depth, 0],
        [half_w - inset, 0, 0],
    ])

    world_verts = joint_center + (
        local_verts[:, 0:1] * tangent
        + local_verts[:, 1:2] * joint_up
        + local_verts[:, 2:3] * joint_normal
    )

    # 榫头 (tail) → A 侧
    tail = _extrude_polygon(world_verts, joint_normal, extrusion_height)

    # 榫槽 (socket) → B 侧（扩大 clearance 为槽）
    socket_verts = joint_center + (
        (local_verts + np.sign(local_verts) * clearance)[:, 0:1] * tangent
        + (local_verts + np.sign(local_verts) * clearance)[:, 1:2] * joint_up
        + local_verts[:, 2:3] * joint_normal
    )
    socket = _extrude_polygon(socket_verts, -joint_normal, extrusion_height * 1.2)

    logger.info("燕尾榫: %.1f×%.1fmm 角度=%d°", tail_width, tail_depth, tail_angle_degrees)
    return tail, socket


# ── 圆柱销 + 孔 ──

def generate_pin_hole_pair(
    position: np.ndarray,
    direction: np.ndarray,
    pin_diameter: float = 4.0,
    pin_length: float = 10.0,
    clearance: float = 0.15,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """生成圆柱销和孔（独立 mesh）."""
    direction = direction / np.linalg.norm(direction)
    radius = pin_diameter / 2

    pin = trimesh.creation.cylinder(radius=radius, height=pin_length, sections=32)
    z_axis = np.array([0.0, 0.0, 1.0])
    if not np.allclose(direction, z_axis):
        rot = _rotation_from_to(z_axis, direction)
        pin.apply_transform(rot)
    pin.apply_translation(position)

    hole_radius = radius + clearance
    hole = trimesh.creation.cylinder(radius=hole_radius, height=pin_length * 1.2, sections=32)
    if not np.allclose(direction, z_axis):
        hole.apply_transform(rot)
    hole.apply_translation(position)

    logger.info("圆柱销: Ø%.1f×%.1fmm", pin_diameter, pin_length)
    return pin, hole


# ── 磁铁槽 ──

def generate_magnet_slot(
    position: np.ndarray,
    face_normal: np.ndarray,
    diameter: float = 6.1,
    depth: float = 3.0,
) -> trimesh.Trimesh:
    """生成磁铁槽（负体积，用于布尔相减）."""
    face_normal = face_normal / np.linalg.norm(face_normal)

    slot = trimesh.creation.cylinder(radius=diameter / 2, height=depth * 2, sections=32)
    z_axis = np.array([0.0, 0.0, 1.0])
    if not np.allclose(face_normal, z_axis):
        rot = _rotation_from_to(z_axis, face_normal)
        slot.apply_transform(rot)
    slot.apply_translation(position - face_normal * depth * 0.5)

    logger.info("磁铁槽: Ø%.1f×%.1fmm", diameter, depth)
    return slot


# ── 卡扣 (Snap-Fit) ──

def generate_snap_fit(
    base_position: np.ndarray,
    engagement_direction: np.ndarray,
    beam_width: float = 4.0,
    beam_length: float = 8.0,
    beam_thickness: float = 1.5,
    hook_height: float = 2.0,
) -> trimesh.Trimesh:
    """生成悬臂卡扣（独立 mesh）."""
    engagement_direction = engagement_direction / np.linalg.norm(engagement_direction)

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, engagement_direction)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    width_dir = np.cross(engagement_direction, up)
    width_dir /= np.linalg.norm(width_dir)

    rot_matrix = np.column_stack([width_dir, engagement_direction, up])

    beam = trimesh.creation.box(extents=[beam_width, beam_length, beam_thickness])
    beam.apply_transform(np.vstack([
        np.column_stack([rot_matrix, base_position + engagement_direction * beam_length * 0.5]),
        [0, 0, 0, 1],
    ]))

    hook = trimesh.creation.box(extents=[beam_width, beam_length * 0.3, hook_height])
    hook_pos = base_position + engagement_direction * (beam_length * 0.8)
    hook.apply_transform(np.vstack([
        np.column_stack([rot_matrix, hook_pos]),
        [0, 0, 0, 1],
    ]))

    snap = trimesh.util.concatenate([beam, hook])
    logger.info("卡扣: %.0f×%.1f×%.1fmm", beam_width, beam_length, beam_thickness)
    return snap


# ── 辅助 ──

def _extrude_polygon(
    vertices: np.ndarray,
    direction: np.ndarray,
    height: float,
) -> trimesh.Trimesh:
    """将多边形沿方向拉伸为三维网格."""
    direction = direction / np.linalg.norm(direction)
    top = vertices + direction * height
    n = len(vertices)

    # 构建底部和顶部的三角剖分（扇形）
    center = vertices.mean(axis=0)
    center_top = center + direction * height
    all_verts = np.vstack([center, vertices, center_top, top])
    # 索引: 0=center, 1..n=bottom_verts, n+1=center_top, n+2..2n+1=top_verts
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([0, i + 1, j + 1])  # 底部
        faces.append([n + 1, n + 2 + j, n + 2 + i])  # 顶部
        # 侧面
        faces.append([i + 1, n + 2 + i, n + 2 + j])
        faces.append([i + 1, n + 2 + j, j + 1])

    result = trimesh.Trimesh(vertices=all_verts, faces=np.array(faces))
    if result.volume < 0:
        result.invert()
    return result


def _rotation_from_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """从 src 旋转到 dst 的 4×4 矩阵."""
    v = np.cross(src, dst)
    c = np.dot(src, dst)
    if np.allclose(v, 0):
        return np.eye(4) if c > 0 else np.diag([-1, -1, -1, 1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    return np.vstack([np.column_stack([r, [0, 0, 0]]), [0, 0, 0, 1]])


def export_connectors(
    parts: list[trimesh.Trimesh],
    output_dir: Path,
) -> dict[str, Path]:
    """导出所有模块及其连接件.

    当前按接触面分析放置位置，生产环境交互式指定.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for i, part in enumerate(parts):
        path = output_dir / f"part_{i:02d}.stl"
        part.export(str(path))
        results[f"part_{i}"] = path
        logger.info("导出: %s (%d 面)", path.name, len(part.faces))

    return results
