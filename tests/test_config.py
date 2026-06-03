"""测试全局配置."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from src.config import Config, ImageConfig, MeshConfig, ReconstructConfig, ServerConfig


def test_image_config_defaults() -> None:
    """ImageConfig 默认值."""
    c = ImageConfig()
    assert c.min_resolution == (512, 512)
    assert c.bg_removal_model == "u2net"


def test_reconstruct_config_defaults() -> None:
    """ReconstructConfig 默认值."""
    c = ReconstructConfig()
    assert c.dust3r_model == "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    assert c.image_size == 512


def test_mesh_config_defaults() -> None:
    """MeshConfig 默认值."""
    c = MeshConfig()
    assert c.target_faces == 500_000
    assert c.voxel_size == 0.2


def test_server_config_defaults() -> None:
    """ServerConfig 默认值."""
    c = ServerConfig()
    assert c.port == 8080
    assert c.max_images_per_task == 60


def test_config_nested() -> None:
    """Config 包含所有子配置."""
    c = Config()
    assert isinstance(c.image, ImageConfig)
    assert isinstance(c.reconstruct, ReconstructConfig)
    assert isinstance(c.mesh, MeshConfig)
    assert isinstance(c.server, ServerConfig)


def test_ensure_dirs_creates_directories() -> None:
    """ensure_dirs 创建所需目录."""
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch("src.config.DATA_DIR", Path(tmp) / "data"):
            with mock.patch("src.config.OUTPUT_DIR", Path(tmp) / "output"):
                with mock.patch("src.config.UPLOAD_DIR", Path(tmp) / "uploads"):
                    c = Config()
                    c.ensure_dirs()
                    assert (Path(tmp) / "data").is_dir()
                    assert (Path(tmp) / "output").is_dir()
                    assert (Path(tmp) / "uploads").is_dir()
