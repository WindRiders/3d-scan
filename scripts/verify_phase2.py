"""验证脚本 — Phase 2 后处理流水线端到端测试."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh as _tm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.postprocess import (
    clean_mesh_full,
    validate_mesh,
    wall_thickness_report,
)
from src.texturing import bake_ambient_occlusion, unwrap_uv
from src.utils import task_id


def main() -> bool:
    tid = task_id()
    work_dir = Path("/tmp/3d-scan-phase2") / tid
    work_dir.mkdir(parents=True)

    errors: list[str] = []

    # ── Step 1: 生成水密测试网格 ──
    print("=" * 60)
    print("Phase 2 后处理流水线验证")
    print("=" * 60)
    print("\n[1/6] 生成水密测试网格...")
    mesh_dir = work_dir / "mesh"
    mesh_dir.mkdir(parents=True)

    # 用 icosphere 做可靠的水密测试网格
    src_mesh = _tm.creation.icosphere(subdivisions=3, radius=10.0)
    # 添加一些噪声模拟真实扫描
    src_mesh.vertices += np.random.randn(*src_mesh.vertices.shape) * 0.05
    ply_path = mesh_dir / "test.ply"
    src_mesh.export(str(ply_path))
    print(f"  ✓ 测试网格: {ply_path.stat().st_size:,} bytes ({len(src_mesh.faces)} 面)")

    # ── Step 2: 网格验证 ──
    print("\n[2/6] 网格质量验证...")
    validation = validate_mesh(ply_path)
    print(f"  watertight={validation.is_watertight} manifold={validation.is_manifold}")
    print(f"  顶点={validation.vertex_count} 面={validation.face_count}")
    print(f"  体积={validation.volume_mm3:.1f}mm³")
    if validation.issues:
        print(f"  问题: {validation.issues}")
    assert validation.face_count > 0, "面数为零"

    # ── Step 3: 完整清理流水线 ──
    print("\n[3/6] 网格清理（去漂浮→修法线→补孔→重网格化）...")
    clean_dir = work_dir / "clean"
    outputs = clean_mesh_full(ply_path, clean_dir)
    final_mesh = outputs["final"]
    print(f"  ✓ 最终网格: {final_mesh.stat().st_size:,} bytes")
    final_val = validate_mesh(final_mesh)
    print(f"  watertight={final_val.is_watertight} faces={final_val.face_count}")
    if not final_val.is_watertight:
        errors.append("清理后网格仍非水密")

    # ── Step 4: UV 展开 ──
    print("\n[4/6] UV 展开...")
    uv_path = work_dir / "uv" / "model_uv.obj"
    uv_path.parent.mkdir(parents=True)
    try:
        result = unwrap_uv(final_mesh, uv_path, tex_resolution=2048)
        print(f"  ✓ UV 展开: {result.stat().st_size:,} bytes")
    except Exception as e:
        errors.append(f"UV 展开失败: {e}")
        print(f"  ✗ {e}")

    # ── Step 5: AO 烘焙 ──
    print("\n[5/6] 环境光遮蔽烘焙...")
    ao_path = work_dir / "ao" / "model_ao.ply"
    ao_path.parent.mkdir(parents=True)
    try:
        bake_ambient_occlusion(final_mesh, ao_path, samples=64)
        print(f"  ✓ AO 烘焙: {ao_path.stat().st_size:,} bytes")
    except Exception as e:
        errors.append(f"AO 烘焙失败: {e}")
        print(f"  ✗ {e}")

    # ── Step 6: 3D 打印分析 ──
    print("\n[6/6] 3D 打印可行性分析...")
    report = wall_thickness_report(final_mesh)
    print(f"  包围盒对角线: {report['bbox_diagonal_mm']}mm")
    print(f"  最小壁厚: {report.get('min_wall_thickness_mm')}mm")
    print(f"  FDM 可打印: {report['fdm_printable']}")
    print(f"  SLA 可打印: {report['sla_printable']}")
    print(f"  悬垂比例: {report['overhang_ratio']:.1%}")
    if report["risk_zones"]:
        print(f"  风险区域: {len(report['risk_zones'])} 个")
        for z in report["risk_zones"]:
            print(f"    - {z['type']} (severity={z['severity']})")

    # ── 结果 ──
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ 验证失败: {len(errors)} 个错误")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("✅ Phase 2 全部验证通过！")
        print(f"  输出目录: {work_dir}")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
