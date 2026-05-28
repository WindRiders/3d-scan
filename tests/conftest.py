"""pytest 全局配置 — 必须在所有 import 之前运行."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
