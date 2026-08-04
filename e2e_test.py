"""Standalone E2E runner for generative-adversarial-rs."""
import subprocess, sys, os, shutil

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(SKILL_DIR, "geoskill-generative-adversarial-rs.py")
PASS = FAIL = 0


def run(args, expect_rc=0, label=""):
    global PASS, FAIL
    r = subprocess.run([sys.executable, SCRIPT] + args,
                       capture_output=True, text=True, timeout=120)
    ok = r.returncode == expect_rc
    PASS += ok; FAIL += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  rc={r.returncode}")
    if not ok:
        print(f"    stderr: {r.stderr[:300]}")


def cleanup(*ds):
    for d in ds:
        p = os.path.join(SKILL_DIR, d)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def main():
    cleanup("_e2e_01", "_e2e_02", "_e2e_03", "_e2e_05")
    print("=" * 60)
    print("generative-adversarial-rs — E2E (5 scenarios)")
    print("=" * 60)
    run(["--bbox", "116", "39", "117", "40", "--output-dir", "_e2e_01"], label="01 基本")
    run(["--bbox", "121", "31", "122", "32", "--output-dir", "_e2e_02", "--quiet"], label="02 上海")
    run(["--bbox", "116.39", "39.90", "116.40", "39.91", "--output-dir", "_e2e_03", "--quiet"], label="03 极小")
    run([], expect_rc=2, label="04 无参数")
    run(["--bbox", "116", "39", "117", "40", "--synthetic", "--output-dir", "_e2e_05", "--quiet"], label="05 合成")
    print("=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    cleanup("_e2e_01", "_e2e_02", "_e2e_03", "_e2e_05")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
