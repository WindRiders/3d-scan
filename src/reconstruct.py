"""三维重建：DUSt3R → 3DGS → Mesh."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from src.config import ReconstructConfig

logger = logging.getLogger(__name__)

for _dust3r_candidate in ["/opt/dust3r", str(Path(__file__).resolve().parents[2] / "dust3r")]:
    if _dust3r_candidate not in sys.path and Path(_dust3r_candidate).is_dir():
        sys.path.insert(0, _dust3r_candidate)

# 国内环境自动使用 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _is_dust3r_available() -> bool:
    try:
        import torch  # noqa: F401
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner  # noqa: F401
        from dust3r.image_pairs import make_pairs  # noqa: F401
        from dust3r.inference import inference  # noqa: F401
        from dust3r.model import AsymmetricCroCo3DStereo  # noqa: F401
        from dust3r.utils.device import to_numpy  # noqa: F401
        from dust3r.utils.image import load_images  # noqa: F401
        return True
    except ImportError:
        return False


def run_dust3r_reconstruction(
    image_paths: list[Path],
    work_dir: Path,
    config: ReconstructConfig,
) -> dict:
    """DUSt3R 多视图重建 → 稠密点云."""
    if not _is_dust3r_available():
        logger.warning("DUSt3R 未安装，使用模拟模式")
        return _simulated_reconstruction(image_paths, work_dir, config)

    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DUSt3R 使用设备: %s", device)

    # 加载模型
    model = AsymmetricCroCo3DStereo.from_pretrained(
        f"naver/{config.dust3r_model}"
    ).to(device)

    # 加载图片 → DUSt3R 格式
    img_paths_str = [str(p.resolve()) for p in image_paths]
    try:
        square_ok = model.square_ok
    except AttributeError:
        square_ok = False
    imgs = load_images(
        img_paths_str, size=512,
        patch_size=model.patch_size, square_ok=square_ok,
    )
    # 单张图片：DUSt3R 至少需要 2 张才能配对，复制一份
    if len(imgs) == 1:
        import copy
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
        logger.info("单张图片模式：已复制为 2 张配对")
    else:
        logger.info("已加载 %d 张图片", len(imgs))

    # 构建图片对 + 推理
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)

    # 全局对齐优化
    mode = (
        GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=300, schedule="linear", lr=0.01,
        )
        logger.info("全局对齐完成, loss=%.4f", loss)

    # 提取稠密点云
    imgs_tensor = to_numpy(scene.imgs)
    pts3d = to_numpy(scene.get_pts3d())
    masks = to_numpy(scene.get_masks())

    # 合并所有视图的可见点 → (N, 6): xyz + rgb
    all_pts = []
    for img, pts, mask in zip(imgs_tensor, pts3d, masks):
        visible = pts[mask]  # (V, 3)
        colors = img[mask]   # (V, 3)
        all_pts.append(np.concatenate([visible, colors], axis=1))

    combined = np.concatenate(all_pts, axis=0).astype(np.float32)
    logger.info("稠密点云: %d 点", combined.shape[0])

    pts_path = work_dir / "pointcloud.npy"
    np.save(pts_path, combined)
    logger.info("点云已保存: %s", pts_path)

    # 保存相机位姿供 3DGS 使用
    cam_poses = to_numpy(scene.get_im_poses())
    cam_path = work_dir / "cameras.npy"
    np.save(cam_path, cam_poses)
    logger.info("相机位姿已保存: %s (%d views)", cam_path, len(cam_poses))

    meta_path = work_dir / "reconstruction_meta.json"
    meta_path.write_text(json.dumps({
        "num_images": len(image_paths),
        "point_count": int(combined.shape[0]),
        "dust3r_model": config.dust3r_model,
    }, indent=2, ensure_ascii=False))

    return {
        "pointcloud": str(pts_path),
        "meta": str(meta_path),
        "cameras": str(cam_path),
    }


def _simulated_reconstruction(
    image_paths: list[Path],
    work_dir: Path,
    config: ReconstructConfig,
) -> dict:
    """模拟重建 — 测试用球面点云."""
    from PIL import Image

    h, w = 0, 0
    for p in image_paths[:1]:
        img = Image.open(p)
        w, h = img.size

    if h == 0:
        h, w = config.image_size, config.image_size

    n_points = 50000
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    r = 1.0 + 0.05 * np.random.randn(n_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    pts_path = work_dir / "pointcloud.npy"
    np.save(pts_path, pts)

    meta_path = work_dir / "reconstruction_meta.json"
    meta_path.write_text(json.dumps({
        "num_images": len(image_paths),
        "point_count": n_points,
        "image_resolution": f"{w}x{h}",
        "mode": "simulated",
    }, indent=2, ensure_ascii=False))

    logger.info("模拟点云: %d 点", n_points)
    return {"pointcloud": str(pts_path), "meta": str(meta_path)}


def run_gaussian_splatting_refinement(
    pointcloud_path: Path,
    image_paths: list[Path],
    work_dir: Path,
    config: ReconstructConfig,
) -> Path:
    """3DGS 细化点云."""
    from src.splatting import run_gaussian_splatting_refinement as gs_refine
    return gs_refine(pointcloud_path, image_paths, work_dir, config)
