"""3D Gaussian Splatting 细化 — 训练循环、损失函数和顶层接口."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ReconstructConfig
from src.gaussian_model import GaussianModel, init_gaussians_from_pointcloud  # noqa: F401
from src.splatting_kernel import _SplatRasterize

logger = logging.getLogger(__name__)


def _build_projection_matrix(focal: float, img_w: int, img_h: int, device: str) -> torch.Tensor:
    """构建 OpenGL 风格投影矩阵."""
    near, far = 0.01, 100.0
    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 2 * focal / img_w
    proj[1, 1] = 2 * focal / img_h
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


def rasterize(
    model: GaussianModel,
    w2c: torch.Tensor,
    focal: float,
    img_h: int,
    img_w: int,
    bg_color: torch.Tensor,
) -> torch.Tensor:
    """渲染 Gaussian 到图像 (逐 Gaussian splat 方式).

    返回: (3, H, W) RGB 图像.
    """
    device = model._xyz.device
    proj = _build_projection_matrix(focal, img_w, img_h, device)

    # 世界坐标 → 相机坐标
    ones = torch.ones(model.num_gaussians, 1, device=device)
    xyz_homo = torch.cat([model._xyz, ones], dim=1)
    xyz_cam = xyz_homo @ w2c.T  # (N, 4)
    xyz_cam = xyz_cam[:, :3]

    # 视锥剔除：z < 0.01 不可见
    z_depth = xyz_cam[:, 2]
    visible = z_depth > 0.01
    if visible.sum() == 0:
        return bg_color.unsqueeze(1).unsqueeze(1).expand(3, img_h, img_w)

    # 投影到 NDC
    xyz_clip = xyz_cam @ proj[:3, :3].T + proj[:3, 3]
    w_clip = xyz_cam @ proj[3, :3] + proj[3, 3]  # actually -z
    w_clip = w_clip + 0.01  # 防止除零
    ndc = xyz_clip / w_clip.unsqueeze(1)

    # NDC → 像素坐标
    means2D = torch.stack(
        [
            (ndc[:, 0] * 0.5 + 0.5) * img_w,
            (ndc[:, 1] * 0.5 + 0.5) * img_h,
        ],
        dim=1,
    )

    # Jacobian: d(screen)/d(camera)
    fx = 0.5 * img_w
    fy = 0.5 * img_h
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], torch.clamp(xyz_cam[:, 2], min=0.01)
    J = torch.zeros(model.num_gaussians, 2, 3, device=device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / (z * z)

    # 3D → 2D 协方差
    cov3D = model.get_covariance()
    W = w2c[:3, :3]
    cov_cam = W @ cov3D @ W.T
    cov2D = J @ cov_cam @ J.transpose(1, 2)
    diag_aug = torch.zeros_like(cov2D)
    diag_aug[:, 0, 0] = 0.3
    diag_aug[:, 1, 1] = 0.3
    cov2D = cov2D + diag_aug

    # 逐 Gaussian splat
    colors = torch.sigmoid(model._features_dc)
    opacities = torch.sigmoid(model._opacity).squeeze(-1)

    rendered = _SplatRasterize.apply(
        means2D,
        cov2D,
        colors,
        opacities,
        z_depth,
        visible,
        img_h,
        img_w,
        bg_color,
        device,
    )
    return rendered


def _load_cameras(camera_path: Path | None, num_views: int, device: str) -> list[dict]:
    """加载相机参数，无文件时生成默认环绕相机."""
    if camera_path is not None and camera_path.exists():
        data = np.load(camera_path)
        # data: (N, 4, 4) world-to-camera 矩阵
        cameras = []
        for i in range(min(len(data), num_views)):
            cameras.append({"w2c": torch.from_numpy(data[i]).float().to(device)})
        return cameras

    # 默认：环绕相机的 world-to-camera（相机绕物体旋转，看向原点）
    logger.warning("无相机参数文件，使用默认环绕视角")
    cameras = []
    for i in range(num_views):
        angle = 2 * np.pi * i / max(num_views, 1)
        radius = 3.0
        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.5])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        w2c = _look_at(eye, center, up)
        cameras.append({"w2c": torch.from_numpy(w2c).float().to(device)})
    return cameras


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """构建 look-at 视图矩阵 (world-to-camera)."""
    f = center - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = f
    M[:3, 3] = -np.array([np.dot(s, eye), np.dot(u, eye), np.dot(f, eye)])
    return M


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """计算 1 - SSIM."""
    C1 = 0.01**2
    C2 = 0.03**2

    # 高斯窗口
    sigma = 1.5
    coords = torch.arange(window_size, device=img1.device, dtype=torch.float32)
    coords -= window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)
    window = window.expand(3, 1, window_size, window_size)  # (3, 1, W, W) for groups=3

    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size // 2, groups=3)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    sigma1 = F.conv2d(img1.unsqueeze(0) ** 2, window, padding=window_size // 2, groups=3) - mu1_sq
    sigma2 = F.conv2d(img2.unsqueeze(0) ** 2, window, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = (
        F.conv2d(
            (img1 * img2).unsqueeze(0),
            window,
            padding=window_size // 2,
            groups=3,
        )
        - mu1 * mu2
    )

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2)
    )
    return 1.0 - ssim_map.mean()


def run_gaussian_splatting_refinement(
    pointcloud_path: Path,
    image_paths: list[Path],
    work_dir: Path,
    config: ReconstructConfig,
) -> Path:
    """3DGS 细化：从 DUSt3R 点云初始化 Gaussian 并多视图优化."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("3DGS 使用设备: %s", device)

    # 加载点云
    pts = np.load(pointcloud_path)
    if pts.shape[1] < 6:
        logger.warning("点云无颜色通道，跳过 3DGS")
        return _placeholder_refine(pointcloud_path, work_dir)

    # 初始化 Gaussian
    model = init_gaussians_from_pointcloud(pts, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)

    # 加载训练图像和相机
    cameras = _load_cameras(work_dir / "cameras.npy", len(image_paths), device)
    bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)
    img_h, img_w = config.image_size, config.image_size

    # 预加载图像
    from PIL import Image

    gt_images = []
    for p in image_paths[: len(cameras)]:
        img = Image.open(p).resize((img_w, img_h))
        gt = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).to(device)
        if gt.shape[-1] == 4:
            gt = gt[..., :3]
        gt_images.append(gt.permute(2, 0, 1))

    iterations = min(config.gaussian_splatting_iterations, len(cameras) * 500)
    densify_interval = max(iterations // 10, 100)
    focal = img_w * 1.2  # ~50° FOV

    for step in range(iterations):
        view_idx = step % len(cameras)
        cam = cameras[view_idx]

        rendered = rasterize(model, cam["w2c"], focal, img_h, img_w, bg_color)
        gt = gt_images[view_idx]

        loss_l1 = F.l1_loss(rendered, gt)
        loss_ssim = ssim_loss(rendered, gt)
        loss = 0.8 * loss_l1 + 0.2 * loss_ssim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 自适应密度控制
        if step > 0 and step % densify_interval == 0:
            model.densify_and_prune(
                max_grad=0.0005,
                min_opacity=0.01,
                max_screen_size=100.0,
            )
            # 重建优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)

        if step % 100 == 0:
            logger.info(
                "3DGS step %d/%d  loss=%.5f  gaussians=%d",
                step,
                iterations,
                loss.item(),
                model.num_gaussians,
            )

    # 提取优化后的点云
    refined_xyz = model._xyz.detach().cpu().numpy()
    refined_colors = torch.sigmoid(model._features_dc).detach().cpu().numpy()

    # 按透明度过滤
    opacity = torch.sigmoid(model._opacity).detach().cpu().numpy().squeeze(-1)
    keep = opacity > 0.1
    if keep.sum() > 1000:
        refined_xyz, refined_colors = refined_xyz[keep], refined_colors[keep]
        logger.info("透明度过滤: %d → %d 点", len(keep), keep.sum())

    refined = np.concatenate([refined_xyz, refined_colors], axis=1).astype(np.float32)
    refined_path = work_dir / "refined_pointcloud.npy"
    np.save(refined_path, refined)
    logger.info("3DGS 细化点云: %s (%d 点)", refined_path, len(refined))
    return refined_path


def _placeholder_refine(pointcloud_path: Path, work_dir: Path) -> Path:
    """无颜色通道时的回退."""
    refined_path = work_dir / "refined_pointcloud.npy"
    pts = np.load(pointcloud_path)
    np.save(refined_path, pts)
    return refined_path
