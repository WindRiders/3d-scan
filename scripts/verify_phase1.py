"""验证脚本 — Phase 1 流水线端到端测试.

使用模拟数据验证：质检 → 去背景 → 重建 → 网格 → STL 导出.
可在本地无 GPU 环境下运行.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.mesh_export import pointcloud_to_mesh, quick_preview_mesh
from src.preprocess import preprocess_images
from src.reconstruct import _simulated_reconstruction
from src.utils import task_id


def generate_test_images(output_dir: Path, count: int = 12) -> list[Path]:
    """生成模拟的多角度图片（纹理背景 + 彩色物体）."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i in range(count):
        angle = i * (360 / count)
        # 带纹理的背景
        arr = np.random.randint(60, 180, (600, 800, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # 画一个带纹理的椭圆代表"物体"
        r = 150
        cx, cy = 400 + int(80 * np.cos(np.radians(angle))), 300
        color = (
            np.random.randint(50, 180),
            np.random.randint(50, 180),
            np.random.randint(50, 180),
        )
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        # 内部添加纹理线条
        for _ in range(30):
            x1 = cx + np.random.randint(-r + 20, r - 20)
            y1 = cy + np.random.randint(-r + 20, r - 20)
            x2 = x1 + np.random.randint(-30, 30)
            y2 = y1 + np.random.randint(-30, 30)
            draw.line([x1, y1, x2, y2], fill=(
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            ))

        p = output_dir / f"test_{i:03d}.jpg"
        img.save(p)
        paths.append(p)

    return paths


def main() -> bool:
    """运行验证。返回 True 表示全部通过."""
    print("=" * 60)
    print("Phase 1 流水线验证")
    print("=" * 60)

    config = Config()
    tid = task_id()
    work_dir = Path("/tmp/3d-scan-verify") / tid
    work_dir.mkdir(parents=True)

    errors: list[str] = []

    # ── Step 1: 生成测试图片 ──
    print("\n[1/5] 生成测试图片...")
    test_imgs = generate_test_images(work_dir / "raw", count=12)
    print(f"  ✓ 生成了 {len(test_imgs)} 张测试图片")

    # ── Step 2: 质检 + 去背景 ──
    print("\n[2/5] 图片预处理（质检 + 去背景）...")
    try:
        clean = preprocess_images(test_imgs, work_dir, config.image)
        print(f"  ✓ 预处理完成: {len(clean)} 张合格图片")
        if len(test_imgs) != len(clean):
            errors.append(f"预处理丢图: {len(clean)}/{len(test_imgs)}")
    except Exception as e:
        errors.append(f"预处理失败: {e}")
        print(f"  ✗ {e}")

    # ── Step 3: 模拟重建 ──
    print("\n[3/5] 三维重建（模拟模式）...")
    try:
        result = _simulated_reconstruction(test_imgs, work_dir, config.reconstruct)
        pcd_path = Path(result["pointcloud"])
        pcd = np.load(pcd_path)
        print(f"  ✓ 点云生成: {pcd.shape[0]} 点, shape={pcd.shape}")
        if pcd.shape[1] != 3:
            errors.append(f"点云维度错误: {pcd.shape}")
    except Exception as e:
        errors.append(f"重建失败: {e}")
        print(f"  ✗ {e}")

    # ── Step 4: 网格导出 ──
    print("\n[4/5] 点云 → 网格 → STL...")
    try:
        ply_path, stl_path = pointcloud_to_mesh(pcd_path, work_dir, config.mesh)
        assert ply_path.exists(), "PLY 文件未生成"
        assert stl_path.exists(), "STL 文件未生成"
        ply_size = ply_path.stat().st_size
        stl_size = stl_path.stat().st_size
        print(f"  ✓ PLY: {ply_size:,} bytes")
        print(f"  ✓ STL: {stl_size:,} bytes")

        if ply_size < 100:
            errors.append(f"PLY 文件过小: {ply_size} bytes")
        if stl_size < 84:  # STL binary header
            errors.append(f"STL 文件过小: {stl_size} bytes")
    except Exception as e:
        errors.append(f"网格导出失败: {e}")
        print(f"  ✗ {e}")

    # ── Step 5: 预览网格 ──
    print("\n[5/5] 快速预览网格...")
    try:
        preview = quick_preview_mesh(pcd_path, work_dir / "preview.ply", voxel_size=0.05)
        assert preview.exists(), "预览文件未生成"
        print(f"  ✓ 预览: {preview.stat().st_size:,} bytes")
    except Exception as e:
        errors.append(f"预览生成失败: {e}")
        print(f"  ✗ {e}")

    # ── 结果 ──
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ 验证失败: {len(errors)} 个错误")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("✅ 全部验证通过！")
        print(f"  输出目录: {work_dir}")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
