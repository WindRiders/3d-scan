"""Web 前端服务测试."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.server import app


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_index_html_content(client: AsyncClient) -> None:
    """首页包含关键 UI 元素."""
    r = await client.get("/")
    assert r.status_code == 200
    html = r.text
    # 核心元素
    assert 'id="drop-zone"' in html
    assert 'id="viewer-canvas"' in html
    assert 'id="btn-upload"' in html
    assert 'id="btn-download-all"' in html
    assert 'id="part-list"' in html
    assert 'id="progress-bar"' in html


@pytest.mark.asyncio
async def test_static_css_served(client: AsyncClient) -> None:
    """CSS 静态文件可访问."""
    r = await client.get("/static/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_static_js_served(client: AsyncClient) -> None:
    """JS 静态文件可访问."""
    r = await client.get("/static/app.js")
    assert r.status_code == 200
    assert "application/javascript" in r.headers.get(
        "content-type", ""
    ) or "text/javascript" in r.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_health_phases(client: AsyncClient) -> None:
    """健康检查返回所有阶段."""
    r = await client.get("/api/health")
    data = r.json()
    assert "1-reconstruct" in data["phases"]
    assert "2-postprocess" in data["phases"]
    assert "3-decompose" in data["phases"]


@pytest.mark.asyncio
async def test_task_limit_enforced(client: AsyncClient) -> None:
    """超过 max_images_per_task 时返回 400."""
    import io

    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    content = buf.getvalue()

    files = [("files", (f"img_{i}.jpg", content, "image/jpeg")) for i in range(61)]
    r = await client.post("/api/tasks", files=files)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_download_nonexistent_file(client: AsyncClient) -> None:
    """下载不存在的任务文件."""
    # 先创建任务
    import io

    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    content = buf.getvalue()

    r = await client.post(
        "/api/tasks",
        files={"files": ("test.jpg", content, "image/jpeg")},
    )
    tid = r.json()["task_id"]
    r = await client.get(f"/api/tasks/{tid}/download/nonexistent.obj")
    assert r.status_code == 404
