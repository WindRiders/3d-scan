"""E2E 测试：多视图合成图像 → DUSt3R → 3DGS → 网格.

在 GPU 服务器上运行:
    cd /opt/3d-scan && HF_ENDPOINT=https://hf-mirror.com python3 scripts/e2e_3dgs_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw


def create_synthetic_views(output_dir: Path, num_views: int = 8) -> list[Path]:
    """生成多角度合成测试图像（彩色球 + 背景纹理）."""
    output_dir.mkdir(parents=True, exist_ok=True)
    size = 512
    paths = []

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        img = Image.new("RGB", (size, size), (240, 240, 245))
        draw = ImageDraw.Draw(img)

        # 背景纹理（棋盘格）
        for y in range(0, size, 64):
            for x in range(0, size, 64):
                if (x // 64 + y // 64) % 2 == 0:
                    draw.rectangle([x, y, x + 63, y + 63], fill=(220, 220, 225))

        # 彩色球体
        cx, cy = size // 2 + int(30 * np.sin(angle)), size // 2
        r = 120
        for dy in range(-r, r):
            for dx in range(-r, r):
                if dx * dx + dy * dy <= r * r:
                    # 球面法线着色
                    z = np.sqrt(max(0, r * r - dx * dx - dy * dy))
                    nx, ny, nz = dx / r, dy / r, z / r
                    # 光照：两个方向光
                    light1 = max(0, nx * 1.0 + nz * 0.5) * 0.7
                    light2 = max(0, -ny * 0.5 + nz * 0.8) * 0.4
                    intensity = min(1.0, light1 + light2 + 0.2)
                    color = (
                        int(200 * intensity * (0.7 + 0.3 * (nx + 1) / 2)),
                        int(150 * intensity),
                        int(220 * intensity * (0.7 + 0.3 * (nz + 1) / 2)),
                    )
                    try:
                        img.putpixel((cx + dx, cy + dy), color)
                    except IndexError:
                        pass

        p = output_dir / f"view_{i:02d}.png"
        img.save(p)
        paths.append(p)
        print(f"  生成视图 {i}: {p.name} (角度 {angle * 180 / np.pi:.0f}°)")

    return paths


def run_e2e(base_url: str, work_dir: Path) -> dict:
    """运行完整 E2E 管线."""
    print("\n=== E2E 3DGS 测试 ===")

    # 1. 生成测试图像
    print("\n[1/5] 生成合成多视图图像...")
    img_paths = create_synthetic_views(work_dir / "test_images", num_views=8)

    # 2. 上传
    print(f"\n[2/5] 上传 {len(img_paths)} 张图像...")
    files = []
    for i, p in enumerate(img_paths):
        files.append(("files", (f"view_{i:02d}.png", open(p, "rb"), "image/png")))
    r = requests.post(f"{base_url}/api/tasks", files=files)
    assert r.status_code == 200, f"上传失败: {r.text}"
    tid = r.json()["task_id"]
    print(f"  任务 ID: {tid}")

    # 3. 处理 (含 3DGS)
    print("\n[3/5] 开始处理（DUSt3R + 3DGS + Mesh）...")
    t0 = time.time()
    r = requests.post(f"{base_url}/api/tasks/{tid}/process?refine_3dgs=1&decompose_parts=0")
    assert r.status_code == 200, f"处理失败: {r.text}"
    elapsed = time.time() - t0
    data = r.json()
    print(f"  处理完成 (耗时 {elapsed:.0f}s)")
    print(f"  状态: {data['status']}")

    # 4. 验证输出结构
    print("\n[4/5] 验证输出结构...")
    output = data.get("output", {})
    checks = {}

    # STL
    assert "stl" in output, "缺少 STL"
    stl_path = Path(output["stl"])
    assert stl_path.exists(), f"STL 不存在: {stl_path}"
    assert stl_path.stat().st_size > 0, "STL 为空"
    checks["stl"] = f"✓ {stl_path.stat().st_size / 1024:.1f} KB"

    # PLY
    assert "ply" in output, "缺少 PLY"
    ply_path = Path(output["ply"])
    assert ply_path.exists(), f"PLY 不存在: {ply_path}"
    checks["ply"] = f"✓ {ply_path.stat().st_size / 1024:.1f} KB"

    # Pointcloud
    assert "pointcloud" in output, "缺少点云"
    pcd_path = Path(output["pointcloud"])
    assert pcd_path.exists(), f"点云不存在: {pcd_path}"
    pcd = np.load(pcd_path)
    checks["pointcloud"] = f"✓ {pcd.shape[0]} 点, shape={pcd.shape}"

    # Refined (3DGS 输出)
    if "refined_pointcloud" in data:
        ref_path = Path(data["refined_pointcloud"])
        if ref_path.exists():
            ref = np.load(ref_path)
            checks["refined"] = f"✓ {ref.shape[0]} 点 (3DGS 优化后)"

    # Print report
    if "print_report" in data:
        rpt = data["print_report"]
        checks["print_report"] = (
            f"✓ bbox={rpt.get('bbox_diagonal_mm')}mm, "
            f"wall={rpt.get('min_wall_thickness_mm')}mm"
        )

    for k, v in checks.items():
        print(f"  {k}: {v}")

    # 5. 下载验证
    print("\n[5/5] 下载验证...")
    for filename, ext in [("model.stl", "stl"), ("final.ply", "ply")]:
        r = requests.get(f"{base_url}/api/tasks/{tid}/download/{filename}")
        if r.status_code == 200:
            out = work_dir / f"downloaded.{ext}"
            out.write_bytes(r.content)
            print(f"  下载 {filename}: {out.stat().st_size / 1024:.1f} KB → {out}")
        else:
            print(f"  下载 {filename}: 失败 ({r.status_code})")

    return {
        "task_id": tid,
        "elapsed_s": elapsed,
        "checks": checks,
        "pointcloud_points": int(pcd.shape[0]),
    }


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    result = run_e2e(base_url, Path("/tmp/e2e_3dgs_test"))
    print(f"\n{'=' * 50}")
    print("E2E 测试通过!")
    print(f"  任务: {result['task_id']}")
    print(f"  耗时: {result['elapsed_s']:.0f}s")
    print(f"  点云: {result['pointcloud_points']} 点")
    print(json.dumps(result["checks"], indent=2, ensure_ascii=False))