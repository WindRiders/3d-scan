# 3D Gaussian Splatting 设计

不依赖 `diff-gaussian-rasterization` CUDA 扩展，用纯 PyTorch 实现可微分 splatting。

## 三文件分工

| 文件 | 行数 | 职责 |
|------|------|------|
| `gaussian_model.py` | 162 | Gaussian 参数模型 + 点云初始化 + 自适应密度控制 |
| `splatting_kernel.py` | 249 | 自定义 autograd Function：前向 splatting + 反向手动梯度 |
| `splatting.py` | 285 | 训练循环 + 光栅化管线 + 损失函数 + 顶层接口 |

## 数据流

```
DUSt3R 点云 (N,6) xyz+rgb
    │
    ▼  init_gaussians_from_pointcloud()
降采样到 50,000 点
    │  xyz → _xyz (Parameter)
    │  rgb → _features_dc (Parameter)
    │  密度估算 → _scaling (Parameter)
    │  单位四元数 → _rotation (Parameter)
    │  inv_sigmoid(0.1) → _opacity (Parameter)
    ▼
GaussianModel (nn.Module)
    │
    ▼  run_gaussian_splatting_refinement()
    │
    ├─► 每轮迭代:
    │   1. 选视角 (环绕相机)
    │   2. rasterize() 渲染当前视角
    │   3. 0.8×L1 + 0.2×(1-SSIM) 损失
    │   4. backward() → optimizer.step()
    │   5. 每 densify_interval 步: densify_and_prune()
    │
    ▼
透明度过滤 (opacity > 0.1)
    │
    ▼
refined_pointcloud.npy (M, 6)
```

## rasterize() 管线

```
GaussianModel 参数
    │
    ▼  get_covariance()
四元数 → 旋转矩阵 R
    │  Σ = RSSᵀRᵀ
    ▼
3D 协方差 Σ (N, 3, 3)
    │
    ▼  world-to-camera × Jacobian
    │  Σ' = J W Σ Wᵀ Jᵀ
    ▼
2D 协方差 Σ' (N, 2, 2)
    │  + 对角线正则化 0.3
    ▼
    │  sigmoid 激活颜色 + 透明度
    ▼
_SplatRasterize.apply()
    │  (自定义 autograd Function)
    ▼
渲染图像 (3, H, W)
```

## _SplatRasterize 自定义 autograd

### 为什么自己写

`diff-gaussian-rasterization` 需要编译 CUDA 扩展，不同 CUDA 版本/GPU 架构容易出现兼容问题。使用纯 PyTorch 的 `torch.autograd.Function` 实现等效功能，零编译依赖。

### 前向传播 (forward)

```
输入: means2D, cov2D, colors, opacities, z_depth, visible
    │
    ▼  按 z_depth 降序排列（远→近）
    │
    ▼  对每个 Gaussian:
    │   1. 计算 2D 协方差特征值 → 确定 splat 范围 (3σ)
    │   2. 裁剪到图像边界
    │   3. 计算马氏距离 (x-μ)ᵀΣ⁻¹(x-μ)
    │   4. 高斯权重 G = exp(-0.5 × mahalanobis)
    │   5. α = opacity × G
    │   6. 累积: C = C + T × α × color
    │      T = T × (1 - α)
    │
    ▼  保存中间变量供反向传播
    │  (means2D, cov2D, inv_cov, T_before, ...)
    ▼
输出: (3, H, W) RGB 图像
```

### 反向传播 (backward)

```
输入: grad_output (dL/dC)
    │
    ▼  按 splat 顺序逆序遍历（近→远）
    │  对每个 Gaussian:
    │   1. 重建前向计算中的 α, G, T_before
    │   2. dL/dα = (Σc dL/dc × c - dL/dT) × T_before
    │   3. dL/d_color = Σ dL/dC × T × α
    │   4. dL/d_opacity = Σ dL/dα × G
    │   5. dL/d_means2D = -Σ dL/d_maha × ∂maha/∂μ
    │   6. dL/d_cov2D = 链式法则通过逆协方差矩阵
    │
    ▼
输出: grad_means2D, grad_cov2D, grad_colors, grad_opacities
```

## 协方差参数化

3D Gaussian 的协方差 Σ = RSSᵀRᵀ，其中：
- **R**: 单位四元数 → 3×3 旋转矩阵
- **S**: diag(s_x, s_y, s_z)，3 个缩放参数

梯度通过 `R` 和 `S` 回传到参数 `_rotation` 和 `_scaling`。

## 自适应密度控制

`densify_and_prune()` 每 10% 训练步数执行一次：

1. **分裂** (高梯度): 梯度范数 > 0.0005 的 Gaussian → 沿最大缩放轴偏移复制
2. **剪除** (低贡献): 透明度 < 0.01 或屏幕半径 > 100px → 删除

分裂后重建 optimizer（Adam 状态与参数维度不兼容）。

## 损失函数

```
L = 0.8 × L1(rendered, gt) + 0.2 × (1 - SSIM(rendered, gt))
```

SSIM 用 11×11 高斯窗口，分组卷积 (groups=3) 对 RGB 各自计算。

## 运行要求

- **GPU**: CUDA 11.8+，PyTorch 2.2+，xformers
- **VRAM**: ~4 GB（50,000 Gaussian，512×512 渲染）
- **训练时间**: 7000 步 ≈ 2-3 分钟 (H20)
- **CPU 回退**: 不可用（无颜色通道时跳过 3DGS，直接使用 DUSt3R 点云）