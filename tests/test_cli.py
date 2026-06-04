"""测试 CLI 命令行入口."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from src.__main__ import main


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    """--help 输出帮助信息."""
    with mock.patch.object(sys, "argv", ["3d-scan", "--help"]):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "serve" in out
    assert "process" in out


def test_cli_serve_help(capsys: pytest.CaptureFixture) -> None:
    """serve 子命令 --help."""
    with mock.patch.object(sys, "argv", ["3d-scan", "serve", "--help"]):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "--port" in out
    assert "--host" in out


def test_cli_process_help(capsys: pytest.CaptureFixture) -> None:
    """process 子命令 --help."""
    with mock.patch.object(sys, "argv", ["3d-scan", "process", "--help"]):
        with pytest.raises(SystemExit):
            main()
    out = capsys.readouterr().out
    assert "input_dir" in out
    assert "--decompose" in out


def test_cli_process_dir_not_found(caplog: pytest.LogCaptureFixture) -> None:
    """process 不存在的目录时退出."""
    with mock.patch.object(sys, "argv", ["3d-scan", "process", "/nonexistent_dir"]):
        with pytest.raises(SystemExit):
            main()
    assert "不存在" in caplog.text