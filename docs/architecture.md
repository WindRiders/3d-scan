# 架构与数据流

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     Web 前端 (HTML/CSS/JS)                │
│                     static/index.html                     │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP (REST)
┌─────────────────────▼───────────────────────────────────┐
│                  FastAPI (server.py)                      │
│  /api/health  /api/tasks  /api/tasks/{id}/process        │
│  /api/tasks/{id}/download  /api/tasks/{id} (DELETE)      │
│                                                          │
│  内存任务状态 (dict)  —  同步处理（无消息队列）             │
└───┬───┬───┬───┬───────┬─────────────────────────────────┘
    │   │   │   │       │
    ▼   ▼   ▼   ▼       ▼
┌───────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│preproc│ │reconstruct│ │mesh_export│ │postprocess│ │decompose │
│  .py  │ │   .py     │ │   .py     │ │   .py     │ │   .py    │
└───┬───┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
    │           │             │             │             │
    ▼           ▼             ▼             ▼             ▼
┌───────┐ ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
│ rembg │ │DUSt3R │     │Open3D │     │Open3D │     │trimesh│
│  cv2  │ │ 3DGS  │     │Poisson│     │Trimesh│     │ numpy │
└───────┘ └───────┘     └───────┘     └───────┘     └───────┘
```

## 数据流

```
图片 (JPG/PNG)                    [uploads/{task_id}/]
    │
    ▼  preprocess.py
质量检查 (清晰度/亮度/分辨率)
    │  rembg 背景移除
    ▼
干净图片 (RGB)                    [output/{task_id}/clean/]
    │
    ▼  reconstruct.py
DUSt3R 多视图推理
    │  全局对齐优化 (PointCloudOptimizer)
    ▼
稠密点云 .npy (xyz+rgb)           [output/{task_id}/pointcloud.npy]
    │
    ├── (可选) splatting.py
    │   3DGS 多视图优化 (L1+SSIM)
    │   自适应密度控制 (分裂/剪除)
    │   透明度过滤
    │   ▼
    │   细化点云 .npy               [output/{task_id}/refined_pointcloud.npy]
    │
    ▼  mesh_export.py
Poisson 表面重建
    │  降采样 + 法线估计
    ▼
网格 .ply / .stl                   [output/{task_id}/model.stl]
    │
    ▼  postprocess.py
TSDF 融合 + 孔洞填充 + 平滑
    │  水密性/流形检查
    │  壁厚分析
    ▼
最终网格 .ply                      [output/{task_id}/clean/final.ply]
    │
    ├── (可选) decompose.py
    │   多视图渲染 → 语义分割
    │   切割面计算 → 笔刷选取
    │   模块导出
    │   ▼
    │   分件 STL                      [output/{task_id}/parts/*.stl]
    │
    ▼  connectors.py (独立模块)
燕尾榫 / 销钉 / 磁铁 / 卡扣
    │
    ▼  texturing.py (独立模块)
UV 展开 + AO/颜色烘焙
```

## 模块依赖关系

```
server.py
  ├── config.py          # 全局配置
  ├── utils.py           # 任务 ID 生成
  ├── preprocess.py ────── cv2, PIL, numpy, rembg
  ├── reconstruct.py ───── torch, dust3r, numpy
  │   └── splatting.py ─── torch, gaussian_model, splatting_kernel
  ├── mesh_export.py ───── open3d, trimesh, numpy
  ├── postprocess.py ───── open3d, trimesh, numpy
  ├── decompose.py ─────── trimesh, numpy
  ├── connectors.py ────── trimesh, numpy
  └── texturing.py ─────── trimesh, numpy, xatlas (可选)
```

## 配置体系

```python
Config                              # 顶层容器
├── ImageConfig                     # 预处理参数
│   ├── min_resolution: (512, 512)
│   ├── max_blur_score: 100.0
│   ├── min/max_brightness: 20/240
│   └── bg_removal_model: "u2net"
├── ReconstructConfig               # 重建参数
│   ├── dust3r_model: "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
│   ├── image_size: 512
│   ├── min_confidence: 3.0
│   ├── gaussian_splatting_iterations: 7000
│   └── mesh_density: 0.5 mm
├── MeshConfig                      # 网格参数
│   ├── target_faces: 500,000
│   ├── hole_fill_radius: 2.0 mm
│   ├── smooth_iterations: 10
│   └── voxel_size: 0.2 mm
└── ServerConfig                    # 服务参数
    ├── host: "0.0.0.0"
    ├── port: 8080
    ├── max_upload_size: 200 MB
    ├── max_images_per_task: 60
    └── result_ttl_seconds: 3600
```

## 关键设计决策

1. **同步处理** — 无 Celery/Redis 消息队列。任务直接在当前请求中处理，简单可靠，适合单 GPU 低频使用场景。

2. **内存任务状态** — 任务记录存储在 `dict` 中，进程重启后丢失。适合原型验证阶段，生产环境需接数据库。

3. **DUSt3R 兼容** — 通过 sys.path 注入加载（无 setup.py），区别处理 `PairViewer`(≤2 图) 和 `PointCloudOptimizer`(>2 图)。

4. **3DGS 自定义 autograd** — 不依赖 diff-gaussian-rasterization，用纯 PyTorch 实现前向 splatting 和反向手动梯度，避免 CUDA 扩展编译问题。

5. **国内网络适配** — `HF_ENDPOINT=https://hf-mirror.com`，pip 清华源，Docker registry 镜像，dust3r 本地拷贝避免 GitHub clone。