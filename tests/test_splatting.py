"""测试 3DGS 模块."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.splatting import (
    GaussianModel,
    _build_projection_matrix,
    _look_at,
    init_gaussians_from_pointcloud,
    rasterize,
    ssim_loss,
)


@pytest.fixture
def sample_pts() -> np.ndarray:
    """生成球面测试点云."""
    n = 500
    phi = np.random.RandomState(42).uniform(0, 2 * np.pi, n)
    theta = np.random.RandomState(42).uniform(0, np.pi, n)
    r = 1.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    colors = np.random.RandomState(42).uniform(0, 1, (n, 3)).astype(np.float32)
    pts = np.concatenate([np.stack([x, y, z], axis=1), colors], axis=1).astype(np.float32)
    return pts


@pytest.fixture
def gaussian_model(sample_pts: np.ndarray) -> GaussianModel:
    return init_gaussians_from_pointcloud(sample_pts, device="cpu")


def test_init_gaussians(sample_pts: np.ndarray) -> None:
    """从点云初始化为有效 Gaussian 参数."""
    model = init_gaussians_from_pointcloud(sample_pts, device="cpu")
    assert model.num_gaussians == 500
    assert model._xyz.shape == (500, 3)
    assert model._features_dc.shape == (500, 3)
    assert model._scaling.shape == (500, 3)
    assert model._rotation.shape == (500, 4)
    assert model._opacity.shape == (500, 1)
    # 缩放值应 > 0
    assert (model._scaling > 0).all()


def test_get_covariance(gaussian_model: GaussianModel) -> None:
    """协方差矩阵为正定对称."""
    cov = gaussian_model.get_covariance()
    assert cov.shape == (500, 3, 3)
    # 检查对称性
    assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-5)
    # 检查特征值 > 0（正定性）
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals > 0).all()


def test_build_projection_matrix() -> None:
    """投影矩阵构建."""
    proj = _build_projection_matrix(focal=500.0, img_w=512, img_h=512, device="cpu")
    assert proj.shape == (4, 4)
    assert proj[0, 0] > 0
    assert proj[1, 1] > 0


def test_look_at() -> None:
    """Look-at 视图矩阵."""
    eye = np.array([2.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    w2c = _look_at(eye, center, up)
    # 检查原点投影到图像中心附近
    origin_cam = w2c @ np.array([0, 0, 0, 1])
    assert origin_cam[2] > 0  # z > 0 (在相机前方)


def test_rasterize(gaussian_model: GaussianModel) -> None:
    """渲染生成有效图像."""
    eye = np.array([2.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    w2c = torch.from_numpy(_look_at(eye, center, up)).float()
    bg = torch.zeros(3)

    img = rasterize(gaussian_model, w2c, focal=500.0, img_h=256, img_w=256, bg_color=bg)
    assert img.shape == (3, 256, 256)
    assert img.min() >= 0
    assert img.max() <= 1.0
    # 至少有些像素被渲染
    assert img.sum() > 0


def test_ssim_loss() -> None:
    """SSIM 损失在 [0, 2] 范围内."""
    img1 = torch.rand(3, 128, 128)
    img2 = torch.rand(3, 128, 128)
    loss = ssim_loss(img1, img2)
    assert 0 <= loss.item() <= 2.0


def test_ssim_loss_identical() -> None:
    """相同图像 SSIM 损失接近 0."""
    img = torch.rand(3, 128, 128)
    loss = ssim_loss(img, img)
    assert loss.item() < 0.1


def test_densify_and_prune(gaussian_model: GaussianModel) -> None:
    """自适应密度控制不崩溃."""
    # 设置假梯度触发分裂
    gaussian_model._xyz.grad = torch.randn_like(gaussian_model._xyz) * 0.001
    gaussian_model.max_radii2D[:] = 10.0
    gaussian_model.densify_and_prune(
        max_grad=0.0001,
        min_opacity=0.005,
        max_screen_size=200.0,
    )
    # 应保持合理的高斯数量（不会全删）
    assert gaussian_model.num_gaussians > 0
