# 3D Scan

多角度图片 → 高精度 3D 模型重建与模块化拆解。

## 管线

```
图片上传 → 预处理(rembg) → DUSt3R 点云重建 → 3DGS 细化(可选)
→ Poisson 网格重建 → 后处理(补洞/去浮/法线) → 语义拆解(可选) → STL/PLY 导出
```

## 项目结构

```
src/
├── server.py            # FastAPI Web 服务入口
├── config.py            # 全局配置
├── preprocess.py        # 图片质量检查 + 背景移除
├── reconstruct.py       # DUSt3R 重建 + 3DGS 细化接口
├── splatting.py         # 3DGS 训练循环 + 损失函数
├── splatting_kernel.py  # 3DGS 自定义 autograd Function
├── gaussian_model.py    # 3D Gaussian 参数模型
├── mesh_export.py       # 点云 → Poisson 网格
├── postprocess.py       # 网格后处理（验证/补洞/清理）
├── decompose.py         # 语义拆解（分割/切分/笔刷/导出）
├── connectors.py        # 连接件生成（燕尾/销钉/磁铁/卡扣）
├── texturing.py         # UV 展开 + AO/颜色烘焙
├── utils.py             # 工具函数
└── web/                 # 前端 (HTML/CSS/JS)
```

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 启动服务
python -m src.server

# 运行测试
pytest -xvs

# 覆盖率
pytest --cov=src --cov-report=term
```

## GPU 测试

部分测试（3DGS、DUSt3R、server E2E）需要 GPU。无 GPU 时自动 skip：

```bash
# 仅 CPU 测试（无需 GPU）
pytest tests/ --ignore=tests/test_reconstruct.py --ignore=tests/test_splatting.py --ignore=tests/test_server.py

# 全部测试（需 GPU）
CUDA_VISIBLE_DEVICES=0 pytest tests/
```

## 许可证

MIT