"""端到端集成测试: 启动服务 → 上传 → 重建 → 下载验证.

仅在 GPU 环境运行 (需要 DUSt3R/CUDA).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image


def _make_multi_view_images(work_dir: Path, num_views: int = 6) -> list[Path]:
    """生成模拟多视角图片: 密集纹理 + 视点偏移."""
    rng = np.random.RandomState(42)
    w, h = 600, 600
    # 密集纹理背景（确保通过模糊检查）
    texture = rng.randint(30, 225, (h * 2, w * 2, 3), dtype=np.uint8)
    # 叠加高频噪声
    noise = rng.randint(-20, 20, (h * 2, w * 2, 3), dtype=np.int16)
    texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # 画大量几何特征（确保 DUSt3R 有足够特征点匹配）
    for _ in range(30):
        cx, cy = rng.randint(0, w * 2), rng.randint(0, h * 2)
        radius = rng.randint(15, 60)
        color = rng.randint(0, 256, 3).tolist()
        y, x = np.ogrid[: h * 2, : w * 2]
        mask = (x - cx) ** 2 + (y - cy) ** 2 < radius**2
        for c in range(3):
            texture[mask, c] = color[c]
    # 画一些矩形特征
    for _ in range(10):
        rx, ry = rng.randint(0, w * 2 - 50), rng.randint(0, h * 2 - 50)
        rw, rh = rng.randint(20, 80), rng.randint(20, 80)
        color = rng.randint(0, 256, 3).tolist()
        texture[ry : ry + rh, rx : rx + rw] = color

    images: list[Path] = []
    shifts = [(0, 0), (40, 0), (-30, 10), (20, -40), (-40, -20), (10, 30)]

    for i, (dx, dy) in enumerate(shifts[:num_views]):
        view = texture[
            max(0, h // 2 + dy) : max(0, h // 2 + dy) + h,
            max(0, w // 2 + dx) : max(0, w // 2 + dx) + w,
        ]
        p = work_dir / f"view_{i:02d}.png"
        Image.fromarray(view).save(p)
        images.append(p)

    return images


@pytest.mark.skipif(not shutil.which("uvicorn"), reason="需要 uvicorn")
class TestEndToEnd:
    """端到端集成测试."""

    @pytest.fixture(scope="class")
    def server_url(self, tmp_path_factory):
        """启动真实 Web 服务."""
        work_dir = tmp_path_factory.mktemp("e2e")
        port = 19876

        env = {
            **__import__("os").environ,
            "DATA_DIR": str(work_dir / "data"),
            "UPLOAD_DIR": str(work_dir / "uploads"),
            "OUTPUT_DIR": str(work_dir / "output"),
        }

        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.server:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        url = f"http://127.0.0.1:{port}"
        # 等待就绪
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                r = httpx.get(f"{url}/api/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            proc.kill()
            pytest.fail("服务器启动超时")

        yield url

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_full_pipeline(self, server_url):
        """完整流水线: 上传多视角图 → 重建 → 下载 STL/PLY."""
        work_dir = Path(__file__).resolve().parent / "e2e_data"
        work_dir.mkdir(exist_ok=True)

        # Step 1: 生成多视角图片
        images = _make_multi_view_images(work_dir, num_views=4)

        # Step 2: 上传
        files = []
        for img in images:
            files.append(("files", (img.name, open(img, "rb"), "image/png")))
        r = httpx.post(f"{server_url}/api/tasks", files=files, timeout=10)
        assert r.status_code == 200, f"上传失败: {r.text}"
        tid = r.json()["task_id"]
        assert len(tid) == 12
        assert r.json()["image_count"] == 4

        # Step 3: 查询
        r = httpx.get(f"{server_url}/api/tasks/{tid}", timeout=5)
        assert r.status_code == 200
        assert r.json()["status"] == "uploaded"

        # Step 4: 处理 (无拆解，快速)
        r = httpx.post(
            f"{server_url}/api/tasks/{tid}/process",
            timeout=300,
        )
        assert r.status_code == 200, f"处理失败: {r.text}"
        data = r.json()
        assert data["status"] == "done", f"状态异常: {data.get('error', 'unknown')}"

        # Step 5: 验证输出文件存在
        output = data["output"]
        assert "stl" in output
        assert "ply" in output
        assert "pointcloud" in output

        # Step 6: 下载 STL
        stl_name = Path(output["stl"]).name
        r = httpx.get(f"{server_url}/api/tasks/{tid}/download/{stl_name}", timeout=10)
        assert r.status_code == 200
        assert len(r.content) > 0
        stl_data = r.content

        # Step 7: 下载 PLY
        ply_name = Path(output["ply"]).name
        r = httpx.get(f"{server_url}/api/tasks/{tid}/download/{ply_name}", timeout=10)
        assert r.status_code == 200
        assert len(r.content) > 0

        # Step 8: STL 文件有效 (ASCII 或 binary)
        import trimesh as _tm
        stl_path = work_dir / f"e2e_{tid}.stl"
        stl_path.write_bytes(stl_data)
        mesh = _tm.load(str(stl_path), force="mesh")
        assert isinstance(mesh, _tm.Trimesh), f"STL 不是有效网格: {type(mesh)}"
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

        # Step 9: 打印报告存在
        assert data.get("print_report") is not None

        # Cleanup: 删除任务
        r = httpx.delete(f"{server_url}/api/tasks/{tid}", timeout=5)
        assert r.status_code == 200

    def test_pipeline_with_decompose(self, server_url):
        """完整流水线 + 拆解."""
        work_dir = Path(__file__).resolve().parent / "e2e_data"
        work_dir.mkdir(exist_ok=True)

        images = _make_multi_view_images(work_dir, num_views=4)

        files = []
        for img in images:
            files.append(("files", (img.name, open(img, "rb"), "image/png")))
        r = httpx.post(f"{server_url}/api/tasks", files=files, timeout=10)
        assert r.status_code == 200
        tid = r.json()["task_id"]

        # 带拆解的处理（简化为 50K 面后再拆 2 个模块）
        r = httpx.post(
            f"{server_url}/api/tasks/{tid}/process?decompose_parts=2",
            timeout=600,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "done"

        # 验证拆解结果
        assert "parts" in data
        assert len(data["parts"]) >= 1
        for part in data["parts"]:
            assert "name" in part
            assert "filename" in part
            assert part["face_count"] > 0

        # 下载拆件
        first_part = data["parts"][0]["filename"]
        r = httpx.get(f"{server_url}/api/tasks/{tid}/download/{first_part}", timeout=10)
        assert r.status_code == 200
        assert len(r.content) > 0

        # Cleanup
        httpx.delete(f"{server_url}/api/tasks/{tid}", timeout=5)

    def test_error_handling(self, server_url):
        """异常流程: 不存在的任务、重复删除."""
        # 404
        r = httpx.get(f"{server_url}/api/tasks/nonexistent", timeout=5)
        assert r.status_code == 404

        r = httpx.post(f"{server_url}/api/tasks/nonexistent/process", timeout=5)
        assert r.status_code == 404

        r = httpx.delete(f"{server_url}/api/tasks/nonexistent", timeout=5)
        assert r.status_code == 404

        # 无文件上传
        r = httpx.post(f"{server_url}/api/tasks", timeout=5)
        assert r.status_code == 422  # FastAPI 验证失败