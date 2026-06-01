"""点云 → 网格 → STL 导出."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from src.config import MeshConfig

logger = logging.getLogger(__name__)


def load_pointcloud(path: Path) -> o3d.geometry.PointCloud:
    """加载点云（支持 .npy / .ply / .xyz，自动识别 xyz+rgb 格式）."""
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        data = np.asarray(o3d.io.read_point_cloud(str(path)).points)

    pcd = o3d.geometry.PointCloud()
    if data.shape[1] >= 6:
        pcd.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6].astype(np.float64))
    else:
        pcd.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))
    return pcd


def remove_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """统计滤波去除离群点."""
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    logger.info("离群点过滤: %d → %d", len(pcd.points), len(ind))
    return pcd.select_by_index(ind)


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.02,
    max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """估计点云法线."""
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
    )
    # 法线一致性定向
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def poisson_mesh(
    pcd: o3d.geometry.PointCloud,
    depth: int = 10,
) -> o3d.geometry.TriangleMesh:
    """Poisson 表面重建."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
    )
    # 裁剪低密度区域
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    verts_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(verts_to_remove)
    logger.info("Poisson 重建完成: %d 面", len(mesh.triangles))
    return mesh


def simplify_mesh(
    mesh: o3d.geometry.TriangleMesh,
    target_faces: int,
) -> o3d.geometry.TriangleMesh:
    """QEM 简化到目标面数."""
    current = len(mesh.triangles)
    if current <= target_faces:
        return mesh
    mesh = mesh.simplify_quadric_decimation(target_faces)
    logger.info("网格简化: %d → %d 面", current, len(mesh.triangles))
    return mesh


def fill_holes(
    mesh: trimesh.Trimesh,
    max_radius: float = 2.0,
) -> trimesh.Trimesh:
    """使用 trimesh 填充孔洞."""
    try:
        trimesh.repair.fill_holes(mesh)
        logger.info("孔洞已填充")
    except Exception:
        logger.debug("孔洞填充跳过（无可用边界）")
    return mesh


def mesh_smooth_laplacian(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int = 10,
) -> o3d.geometry.TriangleMesh:
    """Laplacian 平滑."""
    mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=iterations,
    )
    return mesh


def pointcloud_to_mesh(
    pcd_path: Path,
    output_dir: Path,
    config: MeshConfig,
) -> tuple[Path, Path]:
    """点云 → 完整网格流水线."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = load_pointcloud(pcd_path)
    logger.info("加载点云: %d 点", len(pcd.points))

    pcd = remove_outliers(pcd)
    pcd = estimate_normals(pcd, radius=config.voxel_size * 10)

    # 可选体素降采样（点数过低时自动跳过）
    if config.voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(config.voxel_size)
        if len(pcd_down.points) >= 1000:
            pcd = pcd_down
            logger.info("降采样后: %d 点", len(pcd.points))
        else:
            logger.warning("降采样后仅 %d 点，跳过快采用原始点云", len(pcd_down.points))

    mesh = poisson_mesh(pcd)
    mesh = simplify_mesh(mesh, config.target_faces)
    mesh = mesh_smooth_laplacian(mesh, config.smooth_iterations)

    # 转换为 trimesh 做孔洞填充
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tm = trimesh.Trimesh(vertices=verts, faces=faces)
    tm = fill_holes(tm, config.hole_fill_radius)

    # 写回 open3d
    mesh.vertices = o3d.utility.Vector3dVector(tm.vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(tm.faces.astype(np.int32))
    mesh.compute_vertex_normals()

    ply_path = output_dir / "model.ply"
    o3d.io.write_triangle_mesh(str(ply_path), mesh)
    logger.info("PLY 已保存: %s", ply_path)

    stl_path = output_dir / "model.stl"
    o3d.io.write_triangle_mesh(str(stl_path), mesh)
    logger.info("STL 已保存: %s", stl_path)

    return ply_path, stl_path


def quick_preview_mesh(
    pcd_path: Path,
    output_path: Path,
    voxel_size: float = 1.0,
) -> Path:
    """快速生成低精度预览网格."""
    pcd = load_pointcloud(pcd_path)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd = estimate_normals(pcd, radius=voxel_size * 5)

    # Ball pivoting 比 Poisson 快 10x 以上
    radii = [voxel_size * 2, voxel_size * 4, voxel_size * 8]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii),
    )

    o3d.io.write_triangle_mesh(str(output_path), mesh)
    logger.info("预览网格: %s (%d 面)", output_path, len(mesh.triangles))
    return output_path
