"""E2E tests for generative-adversarial-rs — 5 real-world scenarios."""
import subprocess
import sys
import os
import json

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(SKILL_DIR, "geoskill-generative-adversarial-rs.py")


def run(args, expect_rc=0, timeout=120):
    r = subprocess.run([sys.executable, SCRIPT] + args,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == expect_rc, (
        f"rc={r.returncode} (expected {expect_rc})\n"
        f"stdout={r.stdout[:500]}\nstderr={r.stderr[:500]}")
    return r


def test_01_basic():
    """基本功能：给定 bbox，产出结果 + manifest"""
    out = os.path.join(SKILL_DIR, "_test_out_01")
    run(["--bbox", "116.0", "39.0", "117.0", "40.0", "--output-dir", out])
    assert os.path.isdir(out)
    assert os.path.exists(os.path.join(out, "output-manifest.json"))
    with open(os.path.join(out, "output-manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    assert man["exit_code"] == 0
    assert man["skill"] == "geoskill-generative-adversarial-rs"


def test_02_different_region():
    """不同区域：上海"""
    out = os.path.join(SKILL_DIR, "_test_out_02")
    run(["--bbox", "121.0", "31.0", "122.0", "32.0", "--output-dir", out, "--quiet"])
    assert os.path.isdir(out)


def test_03_edge_tiny():
    """边界情况：极小区域"""
    out = os.path.join(SKILL_DIR, "_test_out_03")
    run(["--bbox", "116.39", "39.90", "116.40", "39.91", "--output-dir", out, "--quiet"])
    assert os.path.isdir(out)


def test_04_error_no_args():
    """错误处理：缺少必要参数 -> exit 2"""
    run([], expect_rc=2)


def test_05_synthetic_mode():
    """合成数据模式（无需网络）"""
    out = os.path.join(SKILL_DIR, "_test_out_05")
    run(["--bbox", "116.0", "39.0", "117.0", "40.0", "--synthetic",
         "--output-dir", out, "--quiet"])
    assert os.path.exists(os.path.join(out, "output-manifest.json"))
