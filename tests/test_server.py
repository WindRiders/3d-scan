"""测试 Web API 端点."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.server import app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _make_test_image(tmp_path: Path) -> Path:
    """生成一张测试图片."""
    arr = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    y, x = np.ogrid[:600, :800]
    mask = ((x - 400) ** 2 + (y - 300) ** 2) < 150**2
    arr[mask] = [100, 150, 200]
    p = tmp_path / "test.jpg"
    Image.fromarray(arr).save(p)
    return p


@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    """健康检查."""
    r = await client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "3-decompose" in data["phases"]


@pytest.mark.asyncio
async def test_create_task(client: AsyncClient, tmp_path: Path) -> None:
    """创建上传任务."""
    img = _make_test_image(tmp_path)
    with open(img, "rb") as f:
        r = await client.post(
            "/api/tasks",
            files={"files": ("test.jpg", f, "image/jpeg")},
        )
    assert r.status_code == 200
    data = r.json()
    assert "task_id" in data
    assert data["image_count"] == 1


@pytest.mark.asyncio
async def test_create_task_multiple(client: AsyncClient, tmp_path: Path) -> None:
    """多图上传."""
    files = []
    for i in range(3):
        img = _make_test_image(tmp_path)
        files.append(("files", (f"img_{i}.jpg", open(img, "rb"), "image/jpeg")))
    r = await client.post("/api/tasks", files=files)
    assert r.status_code == 200
    assert r.json()["image_count"] == 3


@pytest.mark.asyncio
async def test_get_task_not_found(client: AsyncClient) -> None:
    """查询不存在的任务."""
    r = await client.get("/api/tasks/nonexistent")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_task_flow(client: AsyncClient, tmp_path: Path) -> None:
    """完整 API 流程: 创建→查询→处理→下载."""
    # 创建
    img = _make_test_image(tmp_path)
    with open(img, "rb") as f:
        r = await client.post(
            "/api/tasks",
            files={"files": ("test.jpg", f, "image/jpeg")},
        )
    assert r.status_code == 200
    tid = r.json()["task_id"]

    # 查询
    r = await client.get(f"/api/tasks/{tid}")
    assert r.status_code == 200
    assert r.json()["status"] == "uploaded"

    # 处理（无拆解）
    r = await client.post(f"/api/tasks/{tid}/process")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "done"
    assert "output" in data
    assert "stl" in data["output"]

    # 下载 STL
    r = await client.get(f"/api/tasks/{tid}/download/model.stl")
    assert r.status_code == 200
    assert len(r.content) > 0


@pytest.mark.asyncio
async def test_task_flow_with_decompose(client: AsyncClient, tmp_path: Path) -> None:
    """完整流程 + 拆解."""
    img = _make_test_image(tmp_path)
    with open(img, "rb") as f:
        r = await client.post(
            "/api/tasks",
            files={"files": ("test.jpg", f, "image/jpeg")},
        )
    tid = r.json()["task_id"]

    # 处理 + 拆解 3 个模块
    r = await client.post(f"/api/tasks/{tid}/process?decompose_parts=3")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "done"
    assert "parts" in data
    assert len(data["parts"]) >= 1

    # 下载分件
    first_part = data["parts"][0]["filename"]
    r = await client.get(f"/api/tasks/{tid}/download/{first_part}")
    assert r.status_code == 200
    assert len(r.content) > 0


@pytest.mark.asyncio
async def test_delete_task(client: AsyncClient, tmp_path: Path) -> None:
    """删除任务."""
    img = _make_test_image(tmp_path)
    with open(img, "rb") as f:
        r = await client.post(
            "/api/tasks",
            files={"files": ("test.jpg", f, "image/jpeg")},
        )
    tid = r.json()["task_id"]

    r = await client.delete(f"/api/tasks/{tid}")
    assert r.status_code == 200

    r = await client.get(f"/api/tasks/{tid}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_create_task_too_many_images(client: AsyncClient, tmp_path: Path) -> None:
    """上传图片超过限制时返回 400."""
    with mock.patch("src.server.cfg") as m:
        m.server.max_images_per_task = 2
        files = []
        for i in range(3):
            img = _make_test_image(tmp_path)
            files.append(("files", (f"img_{i}.jpg", open(img, "rb"), "image/jpeg")))
        r = await client.post("/api/tasks", files=files)
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_delete_task_not_found(client: AsyncClient) -> None:
    """删除不存在的任务返回 404."""
    r = await client.delete("/api/tasks/nonexistent")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_download_file_task_not_found(client: AsyncClient) -> None:
    """下载时任务不存在返回 404."""
    r = await client.get("/api/tasks/nonexistent/download/model.stl")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_download_file_not_found(client: AsyncClient, tmp_path: Path) -> None:
    """任务存在但文件不存在返回 404."""
    img = _make_test_image(tmp_path)
    with open(img, "rb") as f:
        r = await client.post("/api/tasks", files={"files": ("test.jpg", f, "image/jpeg")})
    tid = r.json()["task_id"]
    r = await client.get(f"/api/tasks/{tid}/download/nonexistent.stl")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_process_task_not_found(client: AsyncClient) -> None:
    """处理不存在的任务返回 404."""
    r = await client.post("/api/tasks/nonexistent/process")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_task_success(client: AsyncClient, tmp_path: Path) -> None:
    """查询已创建的任务返回完整信息."""
    img = _make_test_image(tmp_path)
    with open(img, "rb") as f:
        r = await client.post("/api/tasks", files={"files": ("test.jpg", f, "image/jpeg")})
    tid = r.json()["task_id"]
    r = await client.get(f"/api/tasks/{tid}")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "uploaded"
    assert data["image_count"] == 1


@pytest.mark.asyncio
async def test_static_index(client: AsyncClient) -> None:
    """静态首页可访问."""
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
