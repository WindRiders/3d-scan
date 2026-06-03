# 3D Scan

多角度图片 → 高精度 3D 模型重建与模块化拆解。

## 流水线

```
上传 → 预处理(rembg去背景) → DUSt3R 点云重建 → 3DGS 细化(可选)
    → Poisson 网格 → 后处理(补洞/清理/壁厚) → 拆解(可选) → STL/PLY
```

## 项目结构

```
src/
├── server.py            # FastAPI Web 服务（6 个端点，OpenAPI 文档）
├── config.py            # 全局配置（图片/重建/网格/服务参数）
├── preprocess.py        # 图片质量检查 + rembg 背景移除
├── reconstruct.py       # DUSt3R 多视图重建 + 3DGS 细化
├── splatting.py         # 3DGS 训练循环 + 损失函数
├── splatting_kernel.py  # 3DGS 自定义 autograd Function（前向+反向）
├── gaussian_model.py    # 3D Gaussian 参数模型 + 点云初始化
├── mesh_export.py       # 点云 → Poisson 网格 → STL
├── postprocess.py       # 网格清理、孔洞填充、验证、壁厚分析
├── decompose.py         # 语义拆解（分割/切分/笔刷/导出）
├── connectors.py        # 连接件生成（燕尾榫/销钉/磁铁/卡扣）
├── texturing.py         # UV 展开 + 纹理烘焙
├── utils.py             # 工具函数（ID 生成、点云归一化等）
└── web/static/          # Web 前端（HTML/CSS/JS）
tests/
├── test_server.py       # API 端点（14 个测试，覆盖率 94%）
├── test_preprocess.py   # 图片预处理
├── test_reconstruct.py  # DUSt3R 重建（需 GPU）
├── test_splatting.py    # 3DGS 渲染（需 GPU）
├── test_mesh_export.py  # 网格导出
├── test_postprocess.py  # 后处理
├── test_decompose.py    # 拆解
├── test_connectors.py   # 连接件
├── test_texturing.py    # 纹理
├── test_utils.py        # 工具函数
└── test_web.py          # 前端页面
scripts/
├── ci.sh                # CI 脚本（lint + 测试 + 覆盖率）
├── e2e_3dgs_test.py     # E2E 测试（DUSt3R → 3DGS → 网格）
├── verify_phase1.py     # Phase 1 验证
├── verify_phase2.py     # Phase 2 验证
└── verify_phase3.py     # Phase 3 验证
```

## 快速开始

```bash
pip install -e ".[dev]"
python -m src.server                    # 启动服务 http://localhost:8080
pytest -xvs                            # 运行测试
pytest --cov=src --cov-report=term     # 覆盖率
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/tasks` | 创建任务 + 上传图片 |
| GET | `/api/tasks/{id}` | 查询任务状态 |
| POST | `/api/tasks/{id}/process` | 执行重建流水线 |
| GET | `/api/tasks/{id}/download/{file}` | 下载模型文件 |
| DELETE | `/api/tasks/{id}` | 删除任务和文件 |

启动后访问 `http://localhost:8080/docs` 查看 Swagger 文档。

## 测试覆盖

**94 passed, 4 skipped** | 整体 66%（GPU 模块需 H20 服务器）

| 模块 | 覆盖率 | 说明 |
|------|--------|------|
| config.py | 95% | 配置模型 |
| connectors.py | 97% | 连接件生成 |
| server.py | 94% | API 端点 |
| postprocess.py | 93% | 网格后处理 |
| mesh_export.py | 94% | 网格导出 |
| preprocess.py | 91% | 图片预处理 |
| decompose.py | 91% | 拆解 |
| texturing.py | 90% | 纹理 |
| utils.py | 91% | 工具函数 |
| reconstruct.py | 42% | 需 GPU |
| gaussian_model.py | 0% | 需 GPU |
| splatting.py | 0% | 需 GPU |
| splatting_kernel.py | 0% | 需 GPU |

```bash
# CPU 测试（跳过 GPU 模块）
pytest tests/ --ignore=tests/test_reconstruct.py \
    --ignore=tests/test_splatting.py --ignore=tests/test_server.py

# 全部测试（需 GPU）
CUDA_VISIBLE_DEVICES=0 pytest tests/
```

## Docker 部署

### 构建

```bash
# CPU 版本
docker build -t 3d-scan:latest .

# GPU 版本
docker build --build-arg INSTALL_GPU=1 -t 3d-scan:gpu .
```

### 部署到 GPU 服务器

```bash
docker compose up -d    # 端口 8080，单服务，1 GPU
```

服务自动挂载 `data/`、`output/`、`uploads/` 和 HF 模型缓存到宿主机。

### 远程服务器

- **地址**: 10.200.84.15:8080
- **GPU**: 8× NVIDIA H20 (95 GB VRAM each)
- **基础镜像**: nvidia/cuda:12.1.1-devel-ubuntu22.04
- **PyTorch**: 2.5.1+cu121

## 技术栈

**重建**: DUSt3R (ViTLarge) → 3D Gaussian Splatting → Poisson Surface Reconstruction
**后处理**: Open3D (TSDF/补洞/平滑) + Trimesh (验证/壁厚/水密)
**拆解**: 多视图渲染 + 语义分割 + 切割面 + 笔刷选取
**服务**: FastAPI + Uvicorn，同步处理（无 Celery）
**部署**: Docker + NVIDIA GPU，pip 清华源，HF 镜像

## 许可证

MIT