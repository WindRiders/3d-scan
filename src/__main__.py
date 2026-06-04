"""CLI 入口: python -m src <command>"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("3d-scan")


def cmd_serve(args: argparse.Namespace) -> None:
    """启动 Web 服务."""
    import uvicorn

    logger.info("启动 Web 服务: http://%s:%s", args.host, args.port)
    uvicorn.run(
        "src.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


def cmd_process(args: argparse.Namespace) -> None:
    """批量处理图片目录 → 3D 模型 (直接调用管线，无需启动 Web 服务)."""
    from src.decompose import decompose, export_parts
    from src.mesh_export import pointcloud_to_mesh
    from src.postprocess import clean_mesh_full
    from src.preprocess import preprocess_images
    from src.reconstruct import run_dust3r_reconstruction

    cfg = Config()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error("目录不存在: %s", input_dir)
        sys.exit(1)

    images = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not images:
        logger.error("目录中无图片: %s", input_dir)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/5: 预处理 %d 张图片", len(images))
    clean = preprocess_images(images, output_dir, cfg.image)

    logger.info("Step 2/5: DUSt3R 三维重建")
    recon = run_dust3r_reconstruction(clean, output_dir, cfg.reconstruct)
    pcd_path = Path(recon["pointcloud"])

    logger.info("Step 3/5: 泊松网格")
    ply_path, stl_path = pointcloud_to_mesh(pcd_path, output_dir, cfg.mesh)

    logger.info("Step 4/5: 后处理")
    clean_outputs = clean_mesh_full(ply_path, output_dir / "clean")
    final_mesh = clean_outputs["final"]

    if args.decompose > 0:
        logger.info("Step 5/5: 拆解为 %d 个模块", args.decompose)
        parts = decompose(final_mesh, num_parts=args.decompose, work_dir=output_dir / "parts")
        export_parts(parts, output_dir / "parts")
        logger.info("模块导出完成: %s/parts/", output_dir)
    else:
        logger.info("Step 5/5: 跳过拆解")

    logger.info("完成！STL: %s", stl_path)
    logger.info("完成！PLY: %s", final_mesh)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="3d-scan",
        description="多角度图片 → 高精度 3D 模型重建与模块化拆解",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="启动 Web 服务")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--log-level", default="info")
    p_serve.set_defaults(func=cmd_serve)

    p_proc = sub.add_parser("process", help="批量处理图片目录 → 3D 模型")
    p_proc.add_argument("input_dir", help="图片目录路径")
    p_proc.add_argument("--output", "-o", default="output", help="输出目录")
    p_proc.add_argument("--decompose", type=int, default=0, help="拆解模块数 (0=不拆解)")
    p_proc.set_defaults(func=cmd_process)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
