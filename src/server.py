"""FastAPI Web 服务 — 图片上传、重建、拆解、下载."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import DATA_DIR, OUTPUT_DIR, UPLOAD_DIR
from src.config import config as cfg
from src.decompose import decompose, export_parts
from src.mesh_export import pointcloud_to_mesh
from src.postprocess import clean_mesh_full, wall_thickness_report
from src.preprocess import preprocess_images
from src.reconstruct import run_dust3r_reconstruction, run_gaussian_splatting_refinement
from src.utils import task_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web" / "static"

# ── API 模型 ──

TASK_TAG = "任务管理"
FILES_TAG = "文件下载"


class HealthResponse(BaseModel):
    status: str = Field(description="服务状态")
    phases: list[str] = Field(description="支持的流水线阶段")


class TaskCreateResponse(BaseModel):
    task_id: str = Field(description="任务唯一标识")
    image_count: int = Field(description="已上传图片数量")


class TaskInfo(BaseModel):
    id: str = Field(description="任务 ID")
    status: str = Field(description="当前状态: uploaded/preprocessing/reconstructing/...")
    image_count: int = Field(description="图片数量")
    image_paths: list[str] = Field(description="图片路径列表")


class PartInfo(BaseModel):
    name: str = Field(description="部件名称（中文）")
    filename: str = Field(description="STL 文件名")
    face_count: int = Field(description="面数")


class TaskProcessResponse(BaseModel):
    id: str = Field(description="任务 ID")
    status: str = Field(description="最终状态: done/failed")
    image_count: int = Field(description="图片数量")
    image_paths: list[str] = Field(description="图片路径列表")
    output: dict | None = Field(None, description="输出文件路径 (stl/ply/pointcloud)")
    print_report: dict | None = Field(None, description="3D 打印分析报告")
    parts: list[PartInfo] | None = Field(None, description="拆解部件列表")
    part_files: list[str] | None = Field(None, description="部件文件路径列表")
    error: str | None = Field(None, description="错误信息（status=failed 时）")


class TaskDeleteResponse(BaseModel):
    deleted: str = Field(description="已删除的任务 ID")


app = FastAPI(
    title="3D Scan API",
    description="多角度图片 → 高精度 3D 模型重建与模块化拆解。"
    " 流水线: 上传 → 预处理(去背景) → DUSt3R 重建 → 3DGS 细化(可选)"
    " → Poisson 网格 → 后处理 → 拆解(可选) → STL/PLY 导出。",
    version="0.1.0",
    openapi_tags=[
        {"name": TASK_TAG, "description": "创建、查询、处理、删除重建任务"},
        {"name": FILES_TAG, "description": "下载任务产出的模型文件"},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for d in [DATA_DIR, OUTPUT_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 内存任务状态
tasks: dict = {}


@app.get("/api/health", response_model=HealthResponse, tags=[TASK_TAG], summary="健康检查")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", phases=["1-reconstruct", "2-postprocess", "3-decompose"])


# ── 任务 CRUD ──


@app.post(
    "/api/tasks",
    response_model=TaskCreateResponse,
    tags=[TASK_TAG],
    summary="创建任务并上传图片",
)
async def create_task(files: list[UploadFile] = File(...)):
    if len(files) > cfg.server.max_images_per_task:
        raise HTTPException(400, f"最多 {cfg.server.max_images_per_task} 张")
    tid = task_id()
    task_dir = UPLOAD_DIR / tid
    task_dir.mkdir(parents=True)
    saved: list[Path] = []
    for f in files:
        content = await f.read()
        file_path = task_dir / (f.filename or f"img_{len(saved)}.jpg")
        file_path.write_bytes(content)
        saved.append(file_path)
    tasks[tid] = {
        "id": tid,
        "status": "uploaded",
        "image_count": len(saved),
        "image_paths": [str(p) for p in saved],
    }
    return TaskCreateResponse(task_id=tid, image_count=len(saved))


@app.get("/api/tasks/{tid}", response_model=TaskInfo, tags=[TASK_TAG], summary="查询任务状态")
async def get_task(tid: str):
    t = tasks.get(tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    return t


@app.post(
    "/api/tasks/{tid}/process",
    response_model=TaskProcessResponse,
    tags=[TASK_TAG],
    summary="执行重建流水线",
)
async def process_task(tid: str, decompose_parts: int = 0, refine_3dgs: int = 0):
    t = tasks.get(tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    try:
        image_paths = [Path(p) for p in t["image_paths"]]
        work_dir = OUTPUT_DIR / tid
        work_dir.mkdir(parents=True)

        # Step 1: 预处理
        t["status"] = "preprocessing"
        clean = preprocess_images(image_paths, work_dir, cfg.image)

        # Step 2: 重建
        t["status"] = "reconstructing"
        result = run_dust3r_reconstruction(clean, work_dir, cfg.reconstruct)
        pcd_path = Path(result["pointcloud"])

        # Step 2.5: 3DGS 细化（可选）
        if refine_3dgs > 0:
            t["status"] = "refining_3dgs"
            pcd_path = run_gaussian_splatting_refinement(
                pcd_path,
                clean,
                work_dir,
                cfg.reconstruct,
            )
            t["refined_pointcloud"] = str(pcd_path)

        # Step 3: 网格
        t["status"] = "meshing"
        ply_path, stl_path = pointcloud_to_mesh(pcd_path, work_dir, cfg.mesh)

        # Step 4: 后处理
        t["status"] = "postprocessing"
        clean_outputs = clean_mesh_full(ply_path, work_dir / "clean")
        final_mesh_path = clean_outputs["final"]
        print_report = wall_thickness_report(final_mesh_path)

        t["output"] = {
            "stl": str(stl_path),
            "ply": str(final_mesh_path),
            "pointcloud": str(pcd_path),
        }
        t["print_report"] = print_report

        # Step 5: 拆解（可选）
        if decompose_parts > 0:
            t["status"] = "decomposing"
            import trimesh

            mesh = trimesh.load(str(final_mesh_path), force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                raise HTTPException(500, "网格加载失败")
            # 大网格先简化，避免拆解过慢
            if len(mesh.faces) > 50_000:
                mesh = mesh.simplify_quadric_decimation(50_000)
            decomp = decompose(mesh, method="convexity", num_parts=decompose_parts)
            part_dir = work_dir / "parts"
            part_paths = export_parts(decomp, part_dir, format="stl")

            t["parts"] = [
                {"name": p.name, "filename": f"{p.name}.stl", "face_count": len(p.face_indices)}
                for p in decomp.parts
            ]
            t["part_files"] = [str(p) for p in part_paths]

        t["status"] = "done"
        return t

    except Exception as e:
        logger.exception("任务 %s 失败", tid)
        t["status"] = "failed"
        t["error"] = str(e)
        raise HTTPException(500, str(e))


@app.get("/api/tasks/{tid}/download/{filename:path}", tags=[FILES_TAG], summary="下载模型文件")
async def download_file(tid: str, filename: str):
    t = tasks.get(tid)
    if not t:
        raise HTTPException(404, "任务不存在")

    # 先查 output
    for key, path_str in t.get("output", {}).items():
        p = Path(path_str)
        if p.name == filename and p.exists():
            return FileResponse(p, filename=filename)

    # 再查 part_files
    for path_str in t.get("part_files", []):
        p = Path(path_str)
        if p.name == filename and p.exists():
            return FileResponse(p, filename=filename)

    raise HTTPException(404, "文件不存在")


@app.delete(
    "/api/tasks/{tid}",
    response_model=TaskDeleteResponse,
    tags=[TASK_TAG],
    summary="删除任务和文件",
)
async def delete_task(tid: str):
    t = tasks.pop(tid, None)
    if not t:
        raise HTTPException(404)
    for d in [UPLOAD_DIR / tid, OUTPUT_DIR / tid]:
        if d.exists():
            shutil.rmtree(d)
    return TaskDeleteResponse(deleted=tid)


# ── 静态文件（必须在路由之后） ──


@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
