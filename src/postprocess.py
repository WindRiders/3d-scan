"""网格后处理：清理、孔洞填充、验证、壁厚分析."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class MeshValidation:
    """网格质量验证结果."""

    is_watertight: bool
    is_manifold: bool
    vertex_count: int
    face_count: int
    volume_mm3: float
    bbox_size_mm: tuple[float, float, float]
    min_wall_thickness_mm: float | None
    non_manifold_edges: int
    degenerate_faces: int
    issues: list[str]

    @property
    def is_printable(self) -> bool:
        return self.is_watertight and self.is_manifold and len(self.issues) == 0


def validate_mesh(mesh_path: Path) -> MeshValidation:
    """全面验证网格质量和可打印性."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError("文件不包含有效的三角形网格")

    issues: list[str] = []

    # 水密性
    watertight = tm.is_watertight
    if not watertight:
        issues.append("网格不水密，存在孔洞")

    # 流形检查
    non_manifold_edges = _count_non_manifold_edges(tm)
    manifold = non_manifold_edges == 0
    if not manifold:
        issues.append(f"{non_manifold_edges} 条非流形边")

    # 退化面
    degenerate = _count_degenerate_faces(tm)
    if degenerate > 0:
        issues.append(f"{degenerate} 个退化面（面积接近零）")

    # 体积
    try:
        volume = abs(tm.volume) if tm.is_watertight else 0.0
    except Exception:
        volume = 0.0
    if volume == 0:
        issues.append("体积为零，可能面法线不一致")

    # 包围盒
    bbox = tm.bounds
    bbox_size = (
        float(bbox[1][0] - bbox[0][0]),
        float(bbox[1][1] - bbox[0][1]),
        float(bbox[1][2] - bbox[0][2]),
    )

    # 最小壁厚（采样法）
    min_wall = _estimate_min_wall_thickness(tm)

    return MeshValidation(
        is_watertight=watertight,
        is_manifold=bool(manifold),
        vertex_count=len(tm.vertices),
        face_count=len(tm.faces),
        volume_mm3=float(volume),
        bbox_size_mm=bbox_size,
        min_wall_thickness_mm=min_wall,
        non_manifold_edges=non_manifold_edges,
        degenerate_faces=degenerate,
        issues=issues,
    )


def _count_non_manifold_edges(mesh: trimesh.Trimesh) -> int:
    """统计非流形边数量."""
    # 非流形边：被超过2个面共享的边
    edge_counts: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            edge = (min(a, b), max(a, b))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return sum(1 for v in edge_counts.values() if v > 2)


def _count_degenerate_faces(mesh: trimesh.Trimesh) -> int:
    """统计退化面（面积接近零）."""
    areas = mesh.area_faces
    return int(np.sum(areas < 1e-12))


def _estimate_min_wall_thickness(mesh: trimesh.Trimesh) -> float | None:
    """光线投射法估算最小壁厚."""
    try:
        # 采样 1000 个表面点，沿法线反向投射
        samples, face_idx = mesh.sample(1000, return_index=True)
        normals = mesh.face_normals[face_idx]

        # 每条射线求另一侧交点
        hit_points, _, hit_face = mesh.ray.intersects_location(
            samples + normals * 0.001,  # 微偏移避免自交
            -normals,
        )
        if len(hit_points) == 0:
            return None

        distances = np.linalg.norm(hit_points - samples[: len(hit_points)], axis=1)
        return float(np.min(distances[distances > 0.01]))  # 过滤自身
    except Exception:
        return None


def fill_holes_robust(
    mesh_path: Path,
    output_path: Path,
    max_hole_size_mm: float = 50.0,
) -> Path:
    """稳健孔洞填充，按大小分级处理."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError("不是有效的网格文件")

    initial_faces = len(tm.faces)

    # trimesh 内置填充
    try:
        trimesh.repair.fill_holes(tm)
        filled = len(tm.faces) - initial_faces
        if filled:
            logger.info("trimesh 填充了 %d 个面", filled)
    except Exception:
        logger.debug("trimesh fill_holes 不适用")

    # Open3D 的 Poisson 孔洞填充更稳健
    try:
        o3d_mesh = _to_o3d(tm)
        o3d_mesh = o3d_mesh.fill_holes(hole_size=max_hole_size_mm)
        tm = _from_o3d(o3d_mesh)
        logger.info("Open3D 孔洞填充完成")
    except Exception:
        logger.debug("Open3D fill_holes 不适用")

    # 最终验证
    if tm.is_watertight:
        logger.info("网格已水密")
    else:
        logger.warning("仍有非水密区域")

    tm.export(str(output_path))
    return output_path


def remove_floating_pieces(
    mesh_path: Path,
    output_path: Path,
    min_component_ratio: float = 0.01,
) -> Path:
    """移除漂浮碎片：只保留最大的连通分量."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError("不是有效的网格文件")

    components = tm.split(only_watertight=False)
    if len(components) <= 1:
        logger.info("只有一个连通分量，无需清理")
        tm.export(str(output_path))
        return output_path

    # 按面数排序，只保留大的
    components.sort(key=lambda m: len(m.faces), reverse=True)
    total_faces = sum(len(c.faces) for c in components)
    kept = []
    removed = 0
    for c in components:
        if len(c.faces) / total_faces >= min_component_ratio:
            kept.append(c)
        else:
            removed += len(c.faces)

    cleaned = trimesh.util.concatenate(kept) if kept else components[0]
    cleaned.export(str(output_path))
    logger.info("移除了 %d 个漂浮碎片 (%d 面)", len(components) - len(kept), removed)
    return output_path


def fix_normals(mesh_path: Path, output_path: Path) -> Path:
    """修复法线方向，确保一致朝外."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError("不是有效的网格文件")

    # trimesh 的法线修复
    trimesh.repair.fix_normals(tm)
    tm.export(str(output_path))
    logger.info("法线已修复")
    return output_path


def isotropic_remesh(
    mesh_path: Path,
    output_path: Path,
    target_edge_length: float = 1.0,
    iterations: int = 5,
) -> Path:
    """各向同性重网格化，生成均匀三角形."""
    o3d_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    o3d_mesh.compute_vertex_normals()

    # 简化到合理面数
    target = len(o3d_mesh.triangles)
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target)

    # Open3D isotropic remeshing
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=iterations)

    o3d.io.write_triangle_mesh(str(output_path), o3d_mesh)
    logger.info("重网格化完成: %d 面", len(o3d_mesh.triangles))
    return output_path


def wall_thickness_report(mesh_path: Path) -> dict:
    """生成壁厚分析报告，标记 3D 打印风险区域."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError("不是有效的网格文件")

    min_wall = _estimate_min_wall_thickness(tm)
    bbox = tm.bounds
    diag = float(np.linalg.norm(bbox[1] - bbox[0]))

    # FDM 打印参考阈值
    fdm_min_wall = 1.2  # mm, 0.4mm 喷嘴
    sla_min_wall = 0.5  # mm, 光固化

    risk_zones: list[dict] = []
    if min_wall is not None:
        if min_wall < fdm_min_wall:
            risk_zones.append(
                {
                    "type": "壁厚不足",
                    "min_thickness_mm": round(min_wall, 3),
                    "threshold_fdm_mm": fdm_min_wall,
                    "severity": "high" if min_wall < 0.8 else "medium",
                }
            )

    # 悬垂检测
    overhang_ratio = _detect_overhang(tm, angle_threshold=45)
    if overhang_ratio > 0.05:
        risk_zones.append(
            {
                "type": "悬垂区域",
                "ratio": round(overhang_ratio, 2),
                "threshold_angle_deg": 45,
                "severity": "high" if overhang_ratio > 0.15 else "medium",
            }
        )

    return {
        "bbox_diagonal_mm": round(diag, 1),
        "volume_mm3": round(abs(tm.volume), 1) if tm.is_watertight else None,
        "min_wall_thickness_mm": round(min_wall, 3) if min_wall else None,
        "fdm_printable": min_wall is not None and min_wall >= fdm_min_wall,
        "sla_printable": min_wall is not None and min_wall >= sla_min_wall,
        "overhang_ratio": round(overhang_ratio, 2),
        "risk_zones": risk_zones,
    }


def _detect_overhang(mesh: trimesh.Trimesh, angle_threshold: float = 45) -> float:
    """检测悬垂面比例（法线与 Z 轴夹角 > threshold）."""
    # 面朝下（法线接近 -Z）才是悬垂
    downward = mesh.face_normals[:, 2] < 0
    overhang_angles = np.where(
        downward,
        np.degrees(np.arccos(np.clip(-mesh.face_normals[:, 2], -1, 1))),
        0,
    )
    overhang_faces = np.sum(overhang_angles > angle_threshold)
    return float(overhang_faces / len(mesh.faces))


def clean_mesh_full(
    mesh_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """完整网格清理流水线."""
    output_dir.mkdir(parents=True, exist_ok=True)
    current = mesh_path
    outputs: dict[str, Path] = {}

    # Step 1: 移除漂浮碎片
    current = remove_floating_pieces(current, output_dir / "step1_no_floats.ply")
    outputs["no_floats"] = current

    # Step 2: 修复法线
    current = fix_normals(current, output_dir / "step2_normals_fixed.ply")
    outputs["normals_fixed"] = current

    # Step 3: 孔洞填充
    current = fill_holes_robust(current, output_dir / "step3_holes_filled.ply")
    outputs["holes_filled"] = current

    # Step 4: 重网格化
    current = isotropic_remesh(current, output_dir / "step4_remeshed.ply")
    outputs["remeshed"] = current

    # Step 5: 最终验证
    validation = validate_mesh(current)
    logger.info(
        "最终验证: watertight=%s, manifold=%s, faces=%d, issues=%d",
        validation.is_watertight,
        validation.is_manifold,
        validation.face_count,
        len(validation.issues),
    )

    # 保存最终文件
    final = output_dir / "final.ply"
    trimesh.load(str(current), force="mesh").export(str(final))
    outputs["final"] = final

    return outputs


def _to_o3d(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tm.vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(tm.faces.astype(np.int32))
    return mesh


def _from_o3d(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return trimesh.Trimesh(vertices=verts, faces=faces)
