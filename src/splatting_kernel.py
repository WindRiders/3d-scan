"""3DGS splatting 渲染核心 — 自定义 autograd Function."""

import torch

from src.gaussian_model import SIGMA_CLIP


class _SplatRasterize(torch.autograd.Function):
    """自定义 autograd Function: 前向 no_grad splatting, 反向手动梯度."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        means2D: torch.Tensor,
        cov2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        z_depth: torch.Tensor,
        visible: torch.Tensor,
        img_h: int,
        img_w: int,
        bg_color: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        ctx.img_h = img_h
        ctx.img_w = img_w
        ctx.device = device

        rendered = bg_color.unsqueeze(0).unsqueeze(0).expand(img_h, img_w, 3).clone()
        T = torch.ones(img_h, img_w, 1, device=device)

        order = torch.argsort(z_depth, descending=True)

        a = cov2D[:, 0, 0]
        b = cov2D[:, 0, 1]
        d_val = cov2D[:, 1, 1]
        det = torch.clamp(a * d_val - b * cov2D[:, 1, 0], min=1e-6)
        inv_a = d_val / det
        inv_b = -b / det
        inv_d = a / det

        saved: list[dict] = []

        with torch.no_grad():
            for idx in order:
                if not visible[idx]:
                    continue

                mu = means2D[idx]
                eigval = torch.linalg.eigvalsh(cov2D[idx])
                sigma = torch.sqrt(torch.clamp(eigval, min=1e-6))
                extent = (SIGMA_CLIP * sigma.max()).ceil().long().item()
                extent = max(extent, 3)

                x_min = int(mu[0].item()) - extent
                x_max = int(mu[0].item()) + extent + 1
                y_min = int(mu[1].item()) - extent
                y_max = int(mu[1].item()) + extent + 1

                x_min, x_max = max(0, x_min), min(img_w, x_max)
                y_min, y_max = max(0, y_min), min(img_h, y_max)
                if x_min >= x_max or y_min >= y_max:
                    continue

                yy, xx = torch.meshgrid(
                    torch.arange(y_min, y_max, device=device, dtype=torch.float32),
                    torch.arange(x_min, x_max, device=device, dtype=torch.float32),
                    indexing="ij",
                )
                diff = torch.stack([xx, yy], dim=-1) - mu

                combined = inv_b[idx] + cov2D[idx, 1, 0] / det[idx]
                maha = (
                    inv_a[idx] * diff[..., 0] ** 2
                    + combined * diff[..., 0] * diff[..., 1]
                    + inv_d[idx] * diff[..., 1] ** 2
                )
                gauss = torch.exp(-0.5 * torch.clamp(maha, max=30.0))
                alpha = opacities[idx] * gauss
                alpha_3d = alpha.unsqueeze(-1)

                T_patch = T[y_min:y_max, x_min:x_max].clone()
                rendered[y_min:y_max, x_min:x_max] = (
                    rendered[y_min:y_max, x_min:x_max] + T_patch * alpha_3d * colors[idx]
                )
                T[y_min:y_max, x_min:x_max] = T_patch * (1.0 - alpha_3d)

                saved.append({
                    "idx": idx.item(),
                    "y_min": y_min, "y_max": y_max,
                    "x_min": x_min, "x_max": x_max,
                    "T_before": T_patch,
                })

        ctx.save_for_backward(
            means2D, cov2D, colors, opacities, z_depth, visible,
            inv_a, inv_b, inv_d, det,
        )
        ctx.saved_splat_data = saved
        return rendered.permute(2, 0, 1)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple:
        (
            means2D, cov2D, colors, opacities, z_depth, visible,
            inv_a, inv_b, inv_d, det,
        ) = ctx.saved_tensors
        saved = ctx.saved_splat_data
        img_h, img_w = ctx.img_h, ctx.img_w
        device = ctx.device

        dLdC = grad_output.permute(1, 2, 0).contiguous()
        dLdT = torch.zeros(img_h, img_w, 1, device=device)

        grad_means2D = torch.zeros_like(means2D)
        grad_cov2D = torch.zeros_like(cov2D)
        grad_colors = torch.zeros_like(colors)
        grad_opacities = torch.zeros_like(opacities)

        for data in reversed(saved):
            idx = data["idx"]
            y_min, y_max = data["y_min"], data["y_max"]
            x_min, x_max = data["x_min"], data["x_max"]
            T_before = data["T_before"]

            mu_val = means2D[idx]
            cov_val = cov2D[idx]
            inv_a_val = inv_a[idx]
            inv_b_val = inv_b[idx]
            inv_d_val = inv_d[idx]
            combined_val = inv_b_val + cov_val[1, 0] / det[idx]

            yy, xx = torch.meshgrid(
                torch.arange(y_min, y_max, device=device, dtype=torch.float32),
                torch.arange(x_min, x_max, device=device, dtype=torch.float32),
                indexing="ij",
            )
            diff = torch.stack([xx, yy], dim=-1) - mu_val
            maha = (
                inv_a_val * diff[..., 0] ** 2
                + combined_val * diff[..., 0] * diff[..., 1]
                + inv_d_val * diff[..., 1] ** 2
            )
            gauss = torch.exp(-0.5 * torch.clamp(maha, max=30.0))
            alpha_2d = opacities[idx] * gauss
            alpha_3d = alpha_2d.unsqueeze(-1)

            c_sum = (dLdC[y_min:y_max, x_min:x_max] * colors[idx].detach()).sum(
                dim=-1, keepdim=True
            )
            dl_dalpha_patch = (c_sum - dLdT[y_min:y_max, x_min:x_max]) * T_before

            grad_colors[idx] = (
                dLdC[y_min:y_max, x_min:x_max] * T_before * alpha_3d.detach()
            ).sum(dim=(0, 1))

            dLdT[y_min:y_max, x_min:x_max] = (
                c_sum * alpha_3d.detach()
                + dLdT[y_min:y_max, x_min:x_max] * (1.0 - alpha_3d.detach())
            )

            dl_dalpha_flat = dl_dalpha_patch.squeeze(-1)
            dl_dgauss = dl_dalpha_flat * opacities[idx]

            grad_opacities[idx] = (dl_dalpha_flat * gauss.detach()).sum()

            dgauss_dmaha = -0.5 * gauss.detach()
            dl_dmaha = dl_dgauss * dgauss_dmaha

            dmaha_dx = 2 * inv_a[idx] * diff[..., 0].detach() + (
                inv_b[idx] + cov2D[idx, 1, 0] / det[idx]
            ) * diff[..., 1].detach()
            dmaha_dy = (
                inv_b[idx] + cov2D[idx, 1, 0] / det[idx]
            ) * diff[..., 0].detach() + 2 * inv_d[idx] * diff[..., 1].detach()

            grad_means2D[idx, 0] = -(dl_dmaha * dmaha_dx).sum()
            grad_means2D[idx, 1] = -(dl_dmaha * dmaha_dy).sum()

            det_val = det[idx]
            tmp_cross = cov2D[idx, 1, 0] - cov2D[idx, 0, 1]
            dx2 = diff[..., 0].detach() ** 2
            dxy = diff[..., 0].detach() * diff[..., 1].detach()
            dy2 = diff[..., 1].detach() ** 2

            dl_dinv_a = (dl_dmaha * dx2).sum()
            dl_dcombined = (dl_dmaha * dxy).sum()
            dl_dinv_d = (dl_dmaha * dy2).sum()

            det_sq = det_val * det_val
            a_val, b_val = cov2D[idx, 0, 0], cov2D[idx, 0, 1]
            c_val, d_val2 = cov2D[idx, 1, 0], cov2D[idx, 1, 1]

            grad_cov2D[idx, 0, 0] = (
                dl_dinv_d * (-c_val * d_val2 / det_sq)
                + dl_dinv_a * (-d_val2 ** 2 / det_sq)
                + dl_dcombined * (-d_val2 * tmp_cross / det_sq)
            )
            grad_cov2D[idx, 0, 1] = (
                dl_dinv_d * (a_val * c_val / det_sq)
                + dl_dinv_a * (c_val * d_val2 / det_sq)
                + dl_dcombined * ((-a_val * d_val2 + c_val ** 2) / det_sq)
            )
            grad_cov2D[idx, 1, 0] = (
                dl_dinv_d * (a_val * b_val / det_sq)
                + dl_dinv_a * (b_val * d_val2 / det_sq)
                + dl_dcombined * ((-b_val ** 2 + a_val * d_val2) / det_sq)
            )
            grad_cov2D[idx, 1, 1] = (
                dl_dinv_d * (-a_val ** 2 / det_sq)
                + dl_dinv_a * ((a_val * d_val2 - b_val * c_val - d_val2 ** 2) / det_sq)
                + dl_dcombined * (-a_val * tmp_cross / det_sq)
            )

        return (
            grad_means2D, grad_cov2D, grad_colors, grad_opacities,
            None, None, None, None, None, None,
        )
