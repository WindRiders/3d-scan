# API 使用文档

服务启动后访问 `http://localhost:8080/docs` 查看 Swagger 交互文档。

## 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/tasks` | 创建任务 + 上传图片 |
| GET | `/api/tasks/{id}` | 查询任务状态 |
| POST | `/api/tasks/{id}/process` | 执行重建流水线 |
| GET | `/api/tasks/{id}/download/{file}` | 下载模型文件 |
| DELETE | `/api/tasks/{id}` | 删除任务和所有产出 |

## 完整流程示例

```bash
HOST="http://localhost:8080"

# 1. 上传 3 张多角度图片
TASK=$(curl -s -X POST "$HOST/api/tasks" \
  -F "files=@front.jpg" \
  -F "files=@side.jpg" \
  -F "files=@top.jpg")
TID=$(echo "$TASK" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")
echo "Task: $TID"

# 2. 查询状态
curl -s "$HOST/api/tasks/$TID" | python3 -m json.tool

# 3. 执行重建（无拆解）
curl -s -X POST "$HOST/api/tasks/$TID/process" | python3 -m json.tool

# 4. 下载 STL 模型
curl -O "$HOST/api/tasks/$TID/download/model.stl"

# 5. 清理
curl -s -X DELETE "$HOST/api/tasks/$TID"
```

## 处理参数

`POST /api/tasks/{id}/process` 支持两个可选查询参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `decompose_parts` | 0 | 拆解为 N 个模块（0=不拆解） |
| `refine_3dgs` | 0 | 3DGS 细化迭代次数（0=跳过） |

### 带拆解的处理

```bash
# 拆解为 3 个模块
curl -s -X POST "$HOST/api/tasks/$TID/process?decompose_parts=3" | python3 -m json.tool

# 下载各模块文件
curl -O "$HOST/api/tasks/$TID/download/part_底座.stl"
curl -O "$HOST/api/tasks/$TID/download/part_支架.stl"
curl -O "$HOST/api/tasks/$TID/download/part_连接件.stl"
```

### 带 3DGS 细化的处理

```bash
curl -s -X POST "$HOST/api/tasks/$TID/process?refine_3dgs=7000" | python3 -m json.tool
```

## 响应模型

### 创建任务

```json
{
  "task_id": "a1b2c3d4e5f6",
  "image_count": 3
}
```

### 任务状态

```json
{
  "id": "a1b2c3d4e5f6",
  "status": "uploaded",
  "image_count": 3,
  "image_paths": ["/app/uploads/a1b2c3d4e5f6/front.jpg", "..."]
}
```

状态流转: `uploaded` → `preprocessing` → `reconstructing` → `meshing` → `postprocessing` → `done` / `failed`

### 处理完成

```json
{
  "id": "a1b2c3d4e5f6",
  "status": "done",
  "image_count": 3,
  "image_paths": ["..."],
  "output": {
    "stl": "/app/output/.../model.stl",
    "ply": "/app/output/.../clean/final.ply",
    "pointcloud": "/app/output/.../pointcloud.npy"
  },
  "print_report": {
    "bbox_diagonal_mm": 100.5,
    "volume_mm3": 42000.0,
    "min_wall_thickness_mm": 1.2,
    "fdm_printable": true,
    "sla_printable": false,
    "overhang_ratio": 0.05,
    "risk_zones": []
  },
  "parts": null,
  "part_files": null,
  "error": null
}
```

## 错误码

| 状态码 | 场景 |
|--------|------|
| 400 | 图片数量超过 `max_images_per_task` (60 张) |
| 404 | 任务不存在 / 文件不存在 |
| 500 | 流水线处理失败（详见 `error` 字段） |

## HTTP 调用示例

```python
import httpx

async def pipeline(host: str, image_paths: list[str]):
    async with httpx.AsyncClient(base_url=host) as client:
        # 上传
        files = [("files", (Path(p).name, open(p, "rb"), "image/jpeg"))
                 for p in image_paths]
        r = await client.post("/api/tasks", files=files)
        tid = r.json()["task_id"]

        # 处理
        r = await client.post(f"/api/tasks/{tid}/process")
        result = r.json()
        assert result["status"] == "done"

        # 下载
        stl_url = f"/api/tasks/{tid}/download/model.stl"
        r = await client.get(stl_url)
        Path(f"{tid}.stl").write_bytes(r.content)
```