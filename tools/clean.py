#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python tools/clean.py -y --root "/workspace/Gjj Doc/Code/CAD_TabPFN_1"
预览：python3 clean.py --dry-run
确认后删除：python3 clean.py
无需确认：python3 clean.py -y
指定根目录：python3 clean.py --root path/to/project
'''

import os
import sys
import argparse
import shutil
import fnmatch
import stat

# 目标目录（常见 Python 临时/产物目录）
TEMP_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".nox",
    ".tox",
    ".ipynb_checkpoints",
    "htmlcov",
    ".pyre",
    ".pytype",
    ".benchmarks",
    ".coverage_cache",
    ".pdm-build",
}
# 以这些后缀结尾的目录也清理
TEMP_DIR_SUFFIXES = (".egg-info", ".dist-info", ".eggs")
# 构建产物目录
BUILD_DIRS = {"build", "dist", ".eggs"}

# 目标文件（常见 Python 临时/覆盖率文件）
TEMP_FILES_GLOB = [
    "*.pyc",
    "*.pyo",
    "*$py.class",
    ".coverage",
    ".coverage.*",
]

# 不进入/不清理的目录（安全白名单）
SKIP_DIRS = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "env", "ENV", ".env", ".direnv",
    ".idea", ".vscode",
}

def should_remove_dir(name: str) -> bool:
    if name in TEMP_DIRS or name in BUILD_DIRS:
        return True
    if name.endswith(TEMP_DIR_SUFFIXES):
        return True
    return False

def iter_targets(root: str):
    for cur, dirs, files in os.walk(root, topdown=True):
        # 剪枝：避免进入白名单目录
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        # 目录目标
        for d in list(dirs):
            if should_remove_dir(d):
                full = os.path.join(cur, d)
                yield full, True
                # 不再下钻该目录
                dirs.remove(d)
        # 文件目标
        for pat in TEMP_FILES_GLOB:
            for f in fnmatch.filter(files, pat):
                yield os.path.join(cur, f), False

def rm(path: str, is_dir: bool, dry_run: bool = False):
    if dry_run:
        return
    try:
        def onerror(func, p, exc):
            try:
                os.chmod(p, stat.S_IWUSR)
                func(p)
            except Exception:
                pass
        if is_dir:
            shutil.rmtree(path, onerror=onerror)
        else:
            # 确保可写
            try:
                os.chmod(path, stat.S_IWUSR)
            except Exception:
                pass
            os.remove(path)
    except Exception as e:
        print(f"[warn] failed to remove {path}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(
        description="Clean Python caches and build artifacts under the current directory."
    )
    ap.add_argument("--dry-run", action="store_true", help="只列出将被删除的项目，不实际删除。")
    ap.add_argument("-y", "--yes", action="store_true", help="不进行确认，直接删除。")
    ap.add_argument("--root", default=".", help="根目录，默认当前目录。")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    targets = list(iter_targets(root))

    if not targets:
        print("nothing to clean")
        return

    print(f"found {len(targets)} items")
    for p, is_dir in targets:
        kind = "DIR " if is_dir else "FILE"
        print(f"{kind} {os.path.relpath(p, root)}")

    if args.dry_run:
        return

    if not args.yes:
        ans = input("Remove all listed items? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("aborted")
            return

    removed = 0
    for p, is_dir in targets:
        rm(p, is_dir, dry_run=False)
        removed += 1
    print(f"removed {removed} items")

    
    rm("./test/output", True, dry_run=False)

if __name__ == "__main__":
    main()
