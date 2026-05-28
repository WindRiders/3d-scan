"""GaussianModel — 3D Gaussian 参数模型和点云初始化."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SIGMA_CLIP = 3.0


class GaussianModel(nn.Module):
    """3D Gaussian 参数模型."""

    def __init__(
        self,
        xyz: torch.Tensor,
        features_dc: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
    ):
        super().__init__()
        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter(features_dc)
        self._scaling = nn.Parameter(scaling)
        self._rotation = nn.Parameter(rotation)
        self._opacity = nn.Parameter(opacity)
        self.max_radii2D = torch.zeros(xyz.shape[0], device=xyz.device)

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    def get_covariance(self) -> torch.Tensor:
        """从 scaling + rotation(四元数) 构建 3D 协方差矩阵."""
        rot = self._rotation / self._rotation.norm(dim=1, keepdim=True)
        r, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
        R = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - r*z), 2*(x*z + r*y)], dim=1),
            torch.stack([2*(x*y + r*z), 1 - 2*(x**2 + z**2), 2*(y*z - r*x)], dim=1),
            torch.stack([2*(x*z - r*y), 2*(y*z + r*x), 1 - 2*(x**2 + y**2)], dim=1),
        ], dim=1)
        S = torch.diag_embed(self._scaling)
        cov = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)
        return cov

    def get_params(self) -> dict[str, torch.Tensor]:
        return {
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
        }

    def densify_and_prune(
        self, max_grad: float, min_opacity: float, max_screen_size: float,
    ) -> None:
        """自适应密度控制：分裂高梯度高斯，剪除低透明度和过大的."""
        grads = self._xyz.grad
        if grads is None:
            return
        grad_norm = grads.norm(dim=1)
        opacity = torch.sigmoid(self._opacity).squeeze(-1)

        high_grad = grad_norm > max_grad
        prune_mask = (opacity < min_opacity) | (self.max_radii2D > max_screen_size)

        if high_grad.any():
            self._split_gaussians(high_grad)
        if prune_mask.any():
            self._prune_gaussians(prune_mask)

    def _split_gaussians(self, mask: torch.Tensor) -> None:
        """将选中的高斯分裂为两个（沿最大缩放轴偏移）."""
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return

        for param_name in ["_xyz", "_features_dc", "_opacity"]:
            param = getattr(self, param_name)
            new = param.data[indices].clone()
            setattr(self, param_name, nn.Parameter(torch.cat([param.data, new], dim=0)))

        new_scaling = self._scaling.data[indices].clone() * 0.5
        self._scaling = nn.Parameter(torch.cat([self._scaling.data, new_scaling], dim=0))
        new_rot = self._rotation.data[indices].clone()
        self._rotation = nn.Parameter(torch.cat([self._rotation.data, new_rot], dim=0))

        max_axis = self._scaling.data[-len(indices):].argmax(dim=1)
        offset = torch.zeros_like(self._xyz.data[-len(indices):])
        for j, ax in enumerate(max_axis):
            offset[j, ax] = self._scaling.data[-len(indices) + j, ax]
        self._xyz.data[-len(indices):] += offset

    def _prune_gaussians(self, mask: torch.Tensor) -> None:
        keep = ~mask
        n_before = self.num_gaussians
        for param_name in ["_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"]:
            param = getattr(self, param_name)
            setattr(self, param_name, nn.Parameter(param.data[keep]))
        self.max_radii2D = self.max_radii2D[keep]
        if keep.sum() < n_before:
            logger.info("剪除 %d 个高斯", n_before - keep.sum().item())


def init_gaussians_from_pointcloud(
    pts: np.ndarray,
    device: str = "cuda",
) -> GaussianModel:
    """从 DUSt3R 点云初始化 Gaussian 参数.

    pts: (N, 6) xyz + rgb. 点数过多时自动降采样.
    """
    target = 50000
    if len(pts) > target:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pts), target, replace=False)
        pts = pts[idx]
        logger.info("降采样点云: %d → %d", len(idx), target)

    xyz = torch.from_numpy(pts[:, :3]).float().to(device)
    colors = torch.from_numpy(pts[:, 3:6]).float().to(device)

    extent = xyz.max(dim=0).values - xyz.min(dim=0).values
    density = extent.max() / (xyz.shape[0] ** (1 / 3) * 2)
    scaling = torch.full((xyz.shape[0], 3), float(density), device=device)

    rotation = torch.zeros(xyz.shape[0], 4, device=device)
    rotation[:, 0] = 1.0

    opacity = torch.full((xyz.shape[0], 1), _inv_sigmoid(0.1), device=device)

    logger.info("初始化 %d 个 Gaussian, 初始尺度=%.4f", xyz.shape[0], density)
    return GaussianModel(
        xyz=xyz,
        features_dc=colors,
        scaling=scaling,
        rotation=rotation,
        opacity=opacity,
    )


def _inv_sigmoid(x: float) -> float:
    return np.log(x / (1 - x))
