"""全局配置 — Phase 1 多图重建流水线."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPLOAD_DIR = PROJECT_ROOT / "uploads"


@dataclass
class ImageConfig:
    """图像预处理参数."""
    min_resolution: tuple[int, int] = (512, 512)  # 低于此分辨率拒绝
    max_blur_score: float = 100.0  # Laplacian 方差阈值
    min_brightness: float = 20.0  # 最低平均亮度
    max_brightness: float = 240.0  # 最高平均亮度
    bg_removal_model: str = "u2net"  # rembg 模型


@dataclass
class ReconstructConfig:
    """三维重建参数."""
    dust3r_model: str = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    image_size: int = 512
    min_confidence: float = 3.0  # DUSt3R 置信度阈值
    gaussian_splatting_iterations: int = 7000
    mesh_density: float = 0.5  # mm, 点云采样密度


@dataclass
class MeshConfig:
    """网格后处理参数."""
    target_faces: int = 500_000
    hole_fill_radius: float = 2.0  # mm
    smooth_iterations: int = 10
    voxel_size: float = 0.2  # mm, TSDF 体素尺寸


@dataclass
class ServerConfig:
    """Web 服务配置."""
    host: str = "0.0.0.0"
    port: int = 8080
    max_upload_size: int = 200 * 1024 * 1024  # 200MB
    max_images_per_task: int = 60
    result_ttl_seconds: int = 3600  # 结果文件保留时间


@dataclass
class Config:
    image: ImageConfig = field(default_factory=ImageConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def ensure_dirs(self) -> None:
        for d in [DATA_DIR, OUTPUT_DIR, UPLOAD_DIR]:
            d.mkdir(parents=True, exist_ok=True)


config = Config()
