"""测试 3DGS 模块."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.splatting import (
    GaussianModel,
    _build_projection_matrix,
    _load_cameras,
    _look_at,
    _placeholder_refine,
    init_gaussians_from_pointcloud,
    rasterize,
    run_gaussian_splatting_refinement,
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


def test_get_params(gaussian_model: GaussianModel) -> None:
    """get_params 返回完整参数字典."""
    params = gaussian_model.get_params()
    assert set(params.keys()) == {"xyz", "features_dc", "scaling", "rotation", "opacity"}
    assert params["xyz"].shape == (500, 3)


def test_init_gaussians_downsample() -> None:
    """超量点云自动降采样到 50000."""
    pts = np.random.RandomState(42).uniform(0, 1, (60000, 6)).astype(np.float32)
    model = init_gaussians_from_pointcloud(pts, device="cpu")
    assert model.num_gaussians == 50000


def test_init_gaussians_no_color() -> None:
    """无颜色点云初始化为灰色."""
    pts = np.random.RandomState(42).uniform(0, 1, (100, 3)).astype(np.float32)
    model = init_gaussians_from_pointcloud(pts, device="cpu")
    assert model.num_gaussians == 100


def test_rasterize_no_visible() -> None:
    """所有高斯在相机后方时返回纯背景."""
    # 两个高斯在 Z=-10 处，相机在 Z=10 看向 Z+（背对高斯）
    from src.gaussian_model import init_gaussians_from_pointcloud

    pts = np.array([[0, 0, -10, 1, 0, 0], [0, 0, -10, 0, 1, 0]], dtype=np.float32)
    model = init_gaussians_from_pointcloud(pts, device="cpu")
    eye = np.array([0.0, 0.0, 10.0])
    center = np.array([0.0, 0.0, 20.0])
    up = np.array([0.0, 1.0, 0.0])
    w2c = torch.from_numpy(_look_at(eye, center, up)).float()
    bg = torch.ones(3)
    img = rasterize(model, w2c, focal=500.0, img_h=64, img_w=64, bg_color=bg)
    assert torch.allclose(img, torch.ones(3, 64, 64))


def test_load_cameras_from_file(tmp_path: Path) -> None:
    """从文件加载相机参数."""
    import numpy as np

    camera_file = tmp_path / "cameras.npy"
    # 2 个相机，4×4 矩阵
    cams = np.eye(4)[np.newaxis, :, :].repeat(2, axis=0).astype(np.float32)
    np.save(str(camera_file), cams)
    result = _load_cameras(camera_file, num_views=2, device="cpu")
    assert len(result) == 2
    assert "w2c" in result[0]


def test_load_cameras_default() -> None:
    """无文件时使用默认环绕相机."""
    result = _load_cameras(None, num_views=4, device="cpu")
    assert len(result) == 4
    for cam in result:
        assert "w2c" in cam
        assert cam["w2c"].shape == (4, 4)


def test_placeholder_refine(tmp_path: Path) -> None:
    """无颜色通道时的回退路径."""
    pts = np.random.RandomState(42).uniform(0, 1, (100, 3)).astype(np.float32)
    src = tmp_path / "pts.npy"
    np.save(str(src), pts)
    result = _placeholder_refine(src, tmp_path)
    assert result.exists()
    loaded = np.load(str(result))
    assert loaded.shape == (100, 3)


def test_run_gaussian_splatting_refinement(tmp_path: Path) -> None:
    """完整 3DGS 训练流程."""
    from PIL import Image

    from src.config import ReconstructConfig

    # 合成点云
    pts = np.random.RandomState(42).uniform(-1, 1, (200, 6)).astype(np.float32)
    pts[:, 2] = np.abs(pts[:, 2])  # Z ≥ 0
    pc_path = tmp_path / "pc.npy"
    np.save(str(pc_path), pts)

    # 合成图片
    img_paths = []
    for i in range(2):
        arr = np.random.RandomState(42 + i).randint(0, 255, (128, 128, 3), dtype=np.uint8)
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(str(p))
        img_paths.append(p)

    cfg = ReconstructConfig(
        image_size=128,
        gaussian_splatting_iterations=50,
    )
    result = run_gaussian_splatting_refinement(pc_path, img_paths, tmp_path, cfg)
    assert result.exists()
    refined = np.load(str(result))
    assert refined.shape[0] > 0
    assert refined.shape[1] >= 6


def test_densify_triggers_split(gaussian_model: GaussianModel) -> None:
    """高梯度触发分裂，高斯数量增加."""
    n_before = gaussian_model.num_gaussians
    gaussian_model._xyz.grad = torch.randn_like(gaussian_model._xyz) * 0.1
    gaussian_model.max_radii2D[:] = 1.0
    gaussian_model.densify_and_prune(
        max_grad=0.00001,
        min_opacity=0.005,
        max_screen_size=200.0,
    )
    assert gaussian_model.num_gaussians > n_before


def test_densify_triggers_prune(gaussian_model: GaussianModel) -> None:
    """低透明度触发剪除."""
    gaussian_model._xyz.grad = torch.zeros_like(gaussian_model._xyz)
    gaussian_model.max_radii2D[:] = 10.0
    # 设置低透明度
    gaussian_model._opacity.data[:] = -10.0  # sigmoid(-10) ≈ 0
    n_before = gaussian_model.num_gaussians
    gaussian_model.densify_and_prune(
        max_grad=0.0001,
        min_opacity=0.005,
        max_screen_size=200.0,
    )
    assert gaussian_model.num_gaussians < n_before


def test_densify_no_grad(gaussian_model: GaussianModel) -> None:
    """无梯度时 densify_and_prune 直接返回."""
    gaussian_model._xyz.grad = None
    gaussian_model.densify_and_prune(
        max_grad=0.0001,
        min_opacity=0.005,
        max_screen_size=200.0,
    )
    assert gaussian_model.num_gaussians == 500


def test_run_gaussian_splatting_no_color(tmp_path: Path) -> None:
    """无颜色通道点云触发回退路径."""
    from src.config import ReconstructConfig

    pts = np.random.RandomState(42).uniform(-1, 1, (100, 3)).astype(np.float32)
    pc_path = tmp_path / "pc.npy"
    np.save(str(pc_path), pts)
    cfg = ReconstructConfig(image_size=64, gaussian_splatting_iterations=10)
    result = run_gaussian_splatting_refinement(pc_path, [], tmp_path, cfg)
    assert result.exists()


def test_run_gaussian_splatting_rgba(tmp_path: Path) -> None:
    """RGBA 图像自动剥离 alpha 通道."""
    from PIL import Image

    from src.config import ReconstructConfig

    pts = np.random.RandomState(42).uniform(-1, 1, (100, 6)).astype(np.float32)
    pts[:, 2] = np.abs(pts[:, 2])
    pc_path = tmp_path / "pc.npy"
    np.save(str(pc_path), pts)

    # RGBA 图像
    arr = np.random.RandomState(42).randint(0, 255, (64, 64, 4), dtype=np.uint8)
    img_path = tmp_path / "img.png"
    Image.fromarray(arr).save(str(img_path))

    cfg = ReconstructConfig(image_size=64, gaussian_splatting_iterations=20)
    result = run_gaussian_splatting_refinement(pc_path, [img_path], tmp_path, cfg)
    assert result.exists()


# ===== _SplatRasterize backward 测试 =====


def test_splat_rasterize_backward() -> None:
    """反向传播梯度通过 _SplatRasterize.apply() 正常流动."""
    from src.splatting_kernel import _SplatRasterize

    N = 4
    img_h, img_w = 64, 64

    # means2D: 四个角附近，像素坐标
    means2D = torch.tensor(
        [[20.0, 20.0], [20.0, 44.0], [44.0, 20.0], [44.0, 44.0]],
        requires_grad=True,
    )

    # cov2D: SPD 矩阵，L@L^T + eps*I
    scale = 30.0
    cov_list = [scale * torch.eye(2) for _ in range(N)]
    cov2D = torch.stack(cov_list).requires_grad_(True)

    # colors: 四色
    colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        requires_grad=True,
    )

    # opacities: [0, 1] 范围
    opacities = torch.tensor([0.8, 0.6, 0.7, 0.5], requires_grad=True)

    # z_depth: 递增
    z_depth = torch.tensor([1.0, 2.0, 3.0, 4.0])

    visible = torch.ones(N, dtype=torch.bool)

    bg_color = torch.zeros(3)

    rendered = _SplatRasterize.apply(
        means2D, cov2D, colors, opacities, z_depth, visible,
        img_h, img_w, bg_color, "cpu",
    )

    target = torch.rand(3, img_h, img_w)
    loss = torch.nn.functional.mse_loss(rendered, target)
    loss.backward()

    # 所有四个 require_grad 张量的梯度不应为 None，且应有非零值
    assert means2D.grad is not None, "means2D.grad is None"
    assert means2D.grad.abs().sum() > 0, "means2D.grad is all zeros"

    assert cov2D.grad is not None, "cov2D.grad is None"
    assert cov2D.grad.abs().sum() > 0, "cov2D.grad is all zeros"

    assert colors.grad is not None, "colors.grad is None"
    assert colors.grad.abs().sum() > 0, "colors.grad is all zeros"

    assert opacities.grad is not None, "opacities.grad is None"
    assert opacities.grad.abs().sum() > 0, "opacities.grad is all zeros"


def test_splat_rasterize_invisible_skipped() -> None:
    """不可见高斯被跳过，渲染结果与完全移除该高斯一致."""
    from src.splatting_kernel import _SplatRasterize

    N = 4
    img_h, img_w = 64, 64

    # gaussian 0: 图像中心，覆盖范围大；其余三个在角落
    means2D = torch.tensor(
        [[32.0, 32.0], [16.0, 16.0], [16.0, 48.0], [48.0, 32.0]],
        requires_grad=False,
    )

    cov_list = [
        400.0 * torch.eye(2),   # gaussian 0: 覆盖整张图
        20.0 * torch.eye(2),
        20.0 * torch.eye(2),
        20.0 * torch.eye(2),
    ]
    cov2D = torch.stack(cov_list)

    colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    )
    opacities = torch.tensor([0.9, 0.6, 0.7, 0.5])
    z_depth = torch.tensor([4.0, 1.0, 2.0, 3.0])  # gaussian 0 最近
    bg_color = torch.zeros(3)

    # 1. gaussian 0 不可见
    visible_with_invis = torch.tensor([False, True, True, True])
    out_invis = _SplatRasterize.apply(
        means2D, cov2D, colors, opacities, z_depth, visible_with_invis,
        img_h, img_w, bg_color, "cpu",
    )

    # 2. 完全移除 gaussian 0（只剩 3 个高斯）
    out_without_g0 = _SplatRasterize.apply(
        means2D[1:], cov2D[1:], colors[1:], opacities[1:], z_depth[1:],
        torch.ones(3, dtype=torch.bool),
        img_h, img_w, bg_color, "cpu",
    )

    assert torch.allclose(out_invis, out_without_g0, atol=1e-5), (
        "visible[0]=False 应与完全移除 gaussian 0 的渲染结果一致"
    )
