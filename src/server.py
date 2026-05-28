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

app = FastAPI(title="3D Scan", version="0.1.0")
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


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "phases": ["1-reconstruct", "2-postprocess", "3-decompose"]}


# ── 任务 CRUD ──

@app.post("/api/tasks")
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
        "id": tid, "status": "uploaded",
        "image_count": len(saved),
        "image_paths": [str(p) for p in saved],
    }
    return {"task_id": tid, "image_count": len(saved)}


@app.get("/api/tasks/{tid}")
async def get_task(tid: str):
    t = tasks.get(tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    return t


@app.post("/api/tasks/{tid}/process")
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
                pcd_path, clean, work_dir, cfg.reconstruct,
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


@app.get("/api/tasks/{tid}/download/{filename:path}")
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


@app.delete("/api/tasks/{tid}")
async def delete_task(tid: str):
    t = tasks.pop(tid, None)
    if not t:
        raise HTTPException(404)
    for d in [UPLOAD_DIR / tid, OUTPUT_DIR / tid]:
        if d.exists():
            shutil.rmtree(d)
    return {"deleted": tid}


# ── 静态文件（必须在路由之后） ──

@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
