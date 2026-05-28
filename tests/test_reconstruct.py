"""测试重建模块 (模拟模式)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.config import ReconstructConfig
from src.reconstruct import (
    _is_dust3r_available,
    _simulated_reconstruction,
    run_dust3r_reconstruction,
    run_gaussian_splatting_refinement,
)


@pytest.fixture
def test_images(tmp_path: Path) -> list[Path]:
    """生成 3 张测试图像."""
    paths = []
    rng = np.random.RandomState(42)
    for i in range(3):
        arr = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


@pytest.fixture
def config() -> ReconstructConfig:
    return ReconstructConfig(image_size=512)


class TestSimulatedReconstruction:
    """模拟重建路径测试（本地无 DUSt3R 时走此路径）."""

    def test_produces_pointcloud(self, test_images: list[Path], tmp_path: Path) -> None:
        result = _simulated_reconstruction(test_images, tmp_path, ReconstructConfig())
        pcd_path = Path(result["pointcloud"])
        assert pcd_path.exists()
        pts = np.load(pcd_path)
        assert pts.shape[0] == 50000
        assert pts.shape[1] == 3  # 模拟模式无颜色

    def test_produces_meta(self, test_images: list[Path], tmp_path: Path) -> None:
        result = _simulated_reconstruction(test_images, tmp_path, ReconstructConfig())
        meta_path = Path(result["meta"])
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["num_images"] == 3
        assert meta["mode"] == "simulated"
        assert meta["point_count"] == 50000

    def test_single_image(self, test_images: list[Path], tmp_path: Path) -> None:
        """单张图片也正常运行."""
        result = _simulated_reconstruction(test_images[:1], tmp_path, ReconstructConfig())
        assert Path(result["pointcloud"]).exists()

    def test_pointcloud_is_3d_sphere(self, test_images: list[Path], tmp_path: Path) -> None:
        """模拟点云大致是单位球面分布."""
        result = _simulated_reconstruction(test_images[:2], tmp_path, ReconstructConfig())
        pts = np.load(result["pointcloud"])
        # 各坐标分量最大值接近 1（单位球面）
        assert pts[:, 0].max() < 1.5
        assert pts[:, 0].min() > -1.5


class TestReconstructionWithFallback:
    """DUSt3R 不可用时的回退行为."""

    def test_dust3r_not_available_locally(self) -> None:
        """本地环境 DUSt3R 不可用."""
        if _is_dust3r_available():
            pytest.skip("DUSt3R 可用，跳过多余测试")
        assert not _is_dust3r_available()

    def test_falls_back_to_simulated(
        self, test_images: list[Path], tmp_path: Path,
    ) -> None:
        """DUSt3R 不可用时自动回退到模拟."""
        if _is_dust3r_available():
            pytest.skip("DUSt3R 可用时不测试回退路径")
        result = run_dust3r_reconstruction(test_images, tmp_path, ReconstructConfig())
        assert "pointcloud" in result
        assert Path(result["pointcloud"]).exists()
        meta = json.loads(Path(result["meta"]).read_text())
        assert meta.get("mode") == "simulated"

    def test_many_images(self, tmp_path: Path) -> None:
        """多张图片不出错."""
        if _is_dust3r_available():
            pytest.skip("DUSt3R 可用时跳过大量随机图测试")
        rng = np.random.RandomState(42)
        paths = []
        for i in range(10):
            arr = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            p = tmp_path / f"img_{i}.jpg"
            Image.fromarray(arr).save(p)
            paths.append(p)
        result = run_dust3r_reconstruction(paths, tmp_path, ReconstructConfig())
        assert Path(result["pointcloud"]).exists()


class TestGaussianSplattingRefinement:
    """3DGS 细化委托测试 (需要 torch)."""

    @pytest.fixture(autouse=True)
    def _require_torch(self) -> None:
        pytest.importorskip("torch", reason="需要 torch/splatting 模块")

    def test_delegates_to_splatting_module(
        self, tmp_path: Path,
    ) -> None:
        """run_gaussian_splatting_refinement 委托到 splatting 模块."""
        pts = np.random.RandomState(42).uniform(-1, 1, (1000, 6)).astype(np.float32)
        pcd_path = tmp_path / "test_pcd.npy"
        np.save(pcd_path, pts)

        with patch(
            "src.splatting.run_gaussian_splatting_refinement"
        ) as mock_gs:
            mock_gs.return_value = tmp_path / "refined.npy"
            result = run_gaussian_splatting_refinement(
                pcd_path, [], tmp_path, ReconstructConfig(),
            )
            assert result == tmp_path / "refined.npy"
            mock_gs.assert_called_once()

    def test_with_config_iterations(self, tmp_path: Path) -> None:
        """config 参数被正确传递."""
        pts = np.random.RandomState(42).uniform(-1, 1, (500, 6)).astype(np.float32)
        pcd_path = tmp_path / "pcd.npy"
        np.save(pcd_path, pts)

        config = ReconstructConfig(gaussian_splatting_iterations=1000)
        with patch(
            "src.splatting.run_gaussian_splatting_refinement"
        ) as mock_gs:
            mock_gs.return_value = tmp_path / "ref.npy"
            run_gaussian_splatting_refinement(
                pcd_path, [tmp_path / "a.jpg"], tmp_path, config,
            )
            call_args = mock_gs.call_args[0]
            assert call_args[3].gaussian_splatting_iterations == 1000