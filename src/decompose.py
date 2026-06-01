"""网格模块化拆解：多视图渲染、语义分割、切割面、笔刷选取."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class PartLabel:
    """单个模块标签."""

    part_id: int
    name: str
    face_indices: np.ndarray  # 属于该模块的面索引
    color: tuple[int, int, int]


@dataclass
class Decomposition:
    """拆解结果."""

    original_mesh: trimesh.Trimesh
    parts: list[PartLabel]
    face_labels: np.ndarray  # (n_faces,) 每个面对应的 part_id
    cut_boundaries: list[np.ndarray]  # 切割面上的边索引列表

    @property
    def part_count(self) -> int:
        return len(self.parts)

    def extract_part_mesh(self, part_id: int) -> trimesh.Trimesh:
        """提取单个模块的网格."""
        mask = self.face_labels == part_id
        faces = self.original_mesh.faces[mask]
        # 移除不用的顶点
        used_verts = np.unique(faces)
        vert_map = {old: new for new, old in enumerate(used_verts)}
        new_faces = np.array([[vert_map[v] for v in f] for f in faces])
        new_verts = self.original_mesh.vertices[used_verts]
        return trimesh.Trimesh(vertices=new_verts, faces=new_faces)


# ── 多视图渲染 ──


def render_views(
    mesh: trimesh.Trimesh,
    num_views: int = 12,
    resolution: tuple[int, int] = (512, 512),
) -> list[np.ndarray]:
    """从固定视角渲染网格，返回 RGBA 图像列表."""
    scene = mesh.scene()

    # 经纬球分布视点
    views: list[np.ndarray] = []
    phi = np.linspace(0, np.pi, int(np.sqrt(num_views)) + 2)[1:-1]
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_views)) + 2)[:-1]

    points = []
    for p in phi:
        for t in theta:
            points.append(
                [
                    np.sin(p) * np.cos(t),
                    np.sin(p) * np.sin(t),
                    np.cos(p),
                ]
            )

    for i, direction in enumerate(points[:num_views]):
        img = scene.save_image(resolution=resolution, visible=True)
        views.append(np.asarray(img))

    logger.info("渲染了 %d 个视角 (%dx%d)", len(views), *resolution)
    return views


# ── 语义分割（几何启发式 + 模拟模式） ──


def segment_by_graph_cut(
    mesh: trimesh.Trimesh,
    initial_labels: np.ndarray,
    edge_weight_sigma: float = 0.1,
    iterations: int = 5,
) -> np.ndarray:
    """用 Graph Cut 在大角度二面角处优化分割边界."""
    labels = initial_labels.copy()
    edge_angles = mesh.face_adjacency_angles

    # 构建边→面的映射
    face_adjacency = mesh.face_adjacency
    if len(face_adjacency) == 0:
        return labels

    for _ in range(iterations):
        for idx, (f1, f2) in enumerate(face_adjacency):
            angle = edge_angles[idx]
            if labels[f1] != labels[f2]:
                continue
            # 在大角度边（曲率高处）优先放置切割边界
            if angle > np.radians(20):  # 二面角 > 20°
                # 计算权重：角度越大越倾向于切
                weight = np.exp(-edge_weight_sigma / max(angle, 1e-6))
                if np.random.random() < weight:
                    # 随机分配给相邻标签
                    neighbor_labels = set()
                    for neighbor_idx in mesh.face_adjacency[
                        (face_adjacency[:, 0] == f1) | (face_adjacency[:, 1] == f1)
                    ].flatten():
                        neighbor_labels.add(labels[neighbor_idx])
                    neighbor_labels.discard(labels[f1])
                    if neighbor_labels:
                        labels[f1] = min(neighbor_labels)

    return labels


def segment_by_height(
    mesh: trimesh.Trimesh,
    num_parts: int = 4,
) -> np.ndarray:
    """基于高度的简单分割（Z 轴分层）."""
    centroids = mesh.triangles_center[:, 2]  # Z 坐标
    z_min, z_max = centroids.min(), centroids.max()
    bins = np.linspace(z_min, z_max, num_parts + 1)
    labels = np.digitize(centroids, bins) - 1
    labels = np.clip(labels, 0, num_parts - 1)
    return labels


def segment_by_convexity(
    mesh: trimesh.Trimesh,
    num_parts: int = 4,
) -> np.ndarray:
    """基于凸性分解的近似分割（V-HACD 伪实现）."""
    centroids = mesh.triangles_center
    # 用 KMeans 空间聚类近似
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_parts, random_state=42, n_init=10)
    labels = kmeans.fit_predict(centroids)
    return labels.astype(np.int32)


def segment_semantic(
    mesh: trimesh.Trimesh,
    method: str = "convexity",
    num_parts: int = 4,
    optimize_boundaries: bool = True,
) -> np.ndarray:
    """语义分割入口：选择分割策略并优化边界."""
    if method == "height":
        labels = segment_by_height(mesh, num_parts)
    elif method == "convexity":
        labels = segment_by_convexity(mesh, num_parts)
    else:
        raise ValueError(f"不支持的分割方法: {method}")

    if optimize_boundaries:
        labels = segment_by_graph_cut(mesh, labels)

    logger.info(
        "语义分割完成: %d 个模块, 方法=%s",
        len(np.unique(labels)),
        method,
    )
    return labels


# ── 切割平面工具 ──


@dataclass
class CutPlane:
    """切割平面: point + normal (ax+by+cz+d=0)."""

    point: np.ndarray  # 3D 点
    normal: np.ndarray  # 单位法向量

    @property
    def d(self) -> float:
        return -float(np.dot(self.normal, self.point))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """计算点到平面的符号距离."""
        return np.dot(points, self.normal) + self.d


def cut_mesh_with_plane(
    mesh: trimesh.Trimesh,
    plane: CutPlane,
    fill_holes: bool = True,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """用平面切割网格，返回两个子网格."""
    face_centroids = mesh.triangles_center
    face_dist = plane.signed_distance(face_centroids)

    pos_mask = face_dist > 0
    neg_mask = face_dist < 0

    pos_mesh = _extract_faces(mesh, np.where(pos_mask)[0])
    neg_mesh = _extract_faces(mesh, np.where(neg_mask)[0])

    if fill_holes:
        try:
            trimesh.repair.fill_holes(pos_mesh)
        except Exception:
            pass
        try:
            trimesh.repair.fill_holes(neg_mesh)
        except Exception:
            pass

    return pos_mesh, neg_mesh


def cut_mesh_multi_plane(
    mesh: trimesh.Trimesh,
    planes: list[CutPlane],
) -> list[trimesh.Trimesh]:
    """多平面顺序切割."""
    pieces = [mesh]
    for plane in planes:
        new_pieces = []
        for piece in pieces:
            pos, neg = cut_mesh_with_plane(piece, plane)
            if len(pos.faces) > 0:
                new_pieces.append(pos)
            if len(neg.faces) > 0:
                new_pieces.append(neg)
        pieces = new_pieces
    return pieces


# ── 笔刷选取 ──


def brush_select_faces(
    mesh: trimesh.Trimesh,
    seed_face: int,
    max_angle_degrees: float = 15.0,
    max_faces: int = 5000,
) -> np.ndarray:
    """从种子面开始，按法线相似度区域生长选取."""
    max_angle = np.radians(max_angle_degrees)
    seed_normal = mesh.face_normals[seed_face]

    selected = np.zeros(len(mesh.faces), dtype=bool)
    selected[seed_face] = True
    queue = [seed_face]
    count = 0

    while queue and count < max_faces:
        f = queue.pop(0)
        # 查找相邻面
        neighbors = mesh.face_adjacency[
            (mesh.face_adjacency[:, 0] == f) | (mesh.face_adjacency[:, 1] == f)
        ]
        for adj in neighbors:
            for nf in adj:
                if selected[nf]:
                    continue
                angle = np.arccos(
                    np.clip(np.dot(seed_normal, mesh.face_normals[nf]), -1, 1),
                )
                if angle < max_angle:
                    selected[nf] = True
                    queue.append(nf)
        count += 1

    return np.where(selected)[0]


def brush_select_radius(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """球体范围内选取面."""
    centroids = mesh.triangles_center
    distances = np.linalg.norm(centroids - center, axis=1)
    return np.where(distances < radius)[0]


# ── 完整拆解流水线 ──


def decompose(
    mesh: trimesh.Trimesh,
    method: str = "convexity",
    num_parts: int = 4,
    optimize_boundaries: bool = True,
) -> Decomposition:
    """完整拆解流水线."""
    labels = segment_semantic(
        mesh,
        method=method,
        num_parts=num_parts,
        optimize_boundaries=optimize_boundaries,
    )

    unique_labels = np.unique(labels)
    rng = np.random.RandomState(42)
    parts: list[PartLabel] = []
    part_names = ["底座", "躯干", "头部", "左臂", "右臂", "左腿", "右腿", "配件"]

    for i, lid in enumerate(unique_labels):
        face_idx = np.where(labels == lid)[0]
        parts.append(
            PartLabel(
                part_id=int(lid),
                name=part_names[i] if i < len(part_names) else f"部件{i}",
                face_indices=face_idx,
                color=(
                    int(rng.randint(50, 220)),
                    int(rng.randint(50, 220)),
                    int(rng.randint(50, 220)),
                ),
            )
        )

    # 提取切割边
    boundaries = _extract_cut_boundaries(mesh, labels)

    return Decomposition(
        original_mesh=mesh,
        parts=parts,
        face_labels=labels,
        cut_boundaries=boundaries,
    )


def _extract_faces(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
) -> trimesh.Trimesh:
    """提取子网格."""
    if len(face_indices) == 0:
        return trimesh.Trimesh()
    result = mesh.submesh([face_indices])  # type: ignore[arg-type]
    sub = result[0] if isinstance(result, list) else result
    return sub


def _extract_cut_boundaries(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
) -> list[np.ndarray]:
    """提取不同标签面之间的边界边."""
    boundaries: list[np.ndarray] = []
    for f1, f2 in mesh.face_adjacency:
        if labels[f1] != labels[f2]:
            boundaries.append(np.array([f1, f2]))
    return boundaries


# ── 导出拆解结果 ──


def export_parts(
    decomp: Decomposition,
    output_dir: Path,
    format: str = "stl",
) -> list[Path]:
    """将所有模块分别导出为独立文件."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for part in decomp.parts:
        part_mesh = decomp.extract_part_mesh(part.part_id)
        if len(part_mesh.faces) == 0:
            continue
        path = output_dir / f"{part.name}.{format}"
        part_mesh.export(str(path))
        paths.append(path)
        logger.info("导出: %s (%d 面)", path.name, len(part_mesh.faces))

    return paths
