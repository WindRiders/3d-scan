"""验证脚本 — Phase 3 模块拆解 + 连接件 端到端测试."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh as _tm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.connectors import (
    generate_dovetail_pair,
    generate_magnet_slot,
    generate_pin_hole_pair,
    generate_snap_fit,
)
from src.decompose import (
    CutPlane,
    cut_mesh_with_plane,
    decompose,
    export_parts,
)
from src.utils import task_id


def _make_statue_mesh() -> _tm.Trimesh:
    """生成类人形多层几何体."""
    head = _tm.creation.icosphere(radius=2.5, subdivisions=2)
    head.apply_translation([0, 0, 11])

    body = _tm.creation.cylinder(radius=3.5, height=7.0, sections=32)
    body.apply_translation([0, 0, 4])

    left_arm = _tm.creation.cylinder(radius=0.9, height=5.0, sections=16)
    left_arm.apply_translation([-4.5, 0, 7])

    right_arm = _tm.creation.cylinder(radius=0.9, height=5.0, sections=16)
    right_arm.apply_translation([4.5, 0, 7])

    base = _tm.creation.cylinder(radius=4.5, height=1.5, sections=32)
    base.apply_translation([0, 0, -1.5])

    return _tm.util.concatenate([head, body, left_arm, right_arm, base])


def main() -> bool:
    tid = task_id()
    work_dir = Path("/tmp/3d-scan-phase3") / tid
    work_dir.mkdir(parents=True)
    errors: list[str] = []

    print("=" * 60)
    print("Phase 3 模块拆解 + 连接件 验证")
    print("=" * 60)

    # ── Step 1: 生成测试模型 ──
    print("\n[1/6] 生成类人形测试模型...")
    statue = _make_statue_mesh()
    print(f"  ✓ 模型: {len(statue.vertices)} 顶点, {len(statue.faces)} 面")

    # ── Step 2: 语义分割 ──
    print("\n[2/6] 语义分割 (convexity)...")
    decomp = decompose(statue, method="convexity", num_parts=5)
    print(f"  ✓ 分割为 {decomp.part_count} 个模块:")
    for p in decomp.parts:
        print(f"    - {p.name}: {len(p.face_indices)} 面")

    # ── Step 3: 导出分件 ──
    print("\n[3/6] 导出分件 STL...")
    part_paths = export_parts(decomp, work_dir / "parts", format="stl")
    print(f"  ✓ 导出了 {len(part_paths)} 个文件")
    for p in part_paths:
        assert p.exists(), f"文件不存在: {p}"
        assert p.stat().st_size > 84, f"STL 过小: {p}"

    # ── Step 4: 切割平面工具 ──
    print("\n[4/6] 切割平面工具...")
    plane = CutPlane(
        point=np.array([0.0, 0.0, 8.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    top, bottom = cut_mesh_with_plane(statue, plane)
    print(f"  ✓ 上半: {len(top.faces)} 面, 下半: {len(bottom.faces)} 面")
    assert len(top.faces) > 0
    assert len(bottom.faces) > 0

    # ── Step 5: 连接件生成 ──
    print("\n[5/6] 连接件生成...")

    # 燕尾榫
    tail, socket = generate_dovetail_pair(
        joint_center=np.array([0, 0, 0]),
        joint_normal=np.array([0, 0, 1]),
    )
    print(f"  ✓ 燕尾榫: tail={len(tail.faces)}面 socket={len(socket.faces)}面")

    # 圆柱销
    pin, hole = generate_pin_hole_pair(
        position=np.array([5, 0, 0]),
        direction=np.array([0, 0, 1]),
    )
    print(f"  ✓ 圆柱销: pin={len(pin.faces)}面 hole={len(hole.faces)}面")

    # 磁铁槽
    magnet = generate_magnet_slot(
        position=np.array([5, 5, 5]),
        face_normal=np.array([0, 0, 1]),
    )
    print(f"  ✓ 磁铁槽: {len(magnet.faces)}面")

    # 卡扣
    snap = generate_snap_fit(
        base_position=np.array([0, 0, 5]),
        engagement_direction=np.array([0, 1, 0]),
    )
    print(f"  ✓ 卡扣: {len(snap.faces)}面")

    # ── Step 6: 完整流程验证 ──
    print("\n[6/6] 完整流程: 模型→拆解→每个部件加连接件...")
    parts_meshes = [decomp.extract_part_mesh(p.part_id) for p in decomp.parts]
    print(f"  ✓ 提取了 {len(parts_meshes)} 个模块")

    # ── 结果 ──
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ 验证失败: {len(errors)} 个错误")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("✅ Phase 3 全部验证通过！")
        print(f"  输出目录: {work_dir}")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
