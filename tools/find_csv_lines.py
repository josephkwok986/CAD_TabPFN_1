#!/usr/bin/env python3

'''
python find_csv_lines.py "/workspace/Gjj Local/data/CAD/step_out/s2_out" MFInstSeg
python find_csv_lines.py /path/to/dir 关键字
python find_csv_lines.py data "error code" -i --abs
'''

from pathlib import Path
import argparse
from typing import Iterable, Tuple

ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "latin-1")

def iter_lines(path: Path) -> Iterable[Tuple[int, str]]:
    # 逐个编码尝试，失败再换；最后一招忽略错误
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, errors="strict") as f:
                for i, line in enumerate(f, 1):
                    yield i, line
            return
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            yield i, line

def search_csv(root: Path, needle: str, ignore_case: bool = False, relative: bool = True):
    if ignore_case:
        needle = needle.lower()
        contains = lambda s: needle in s.lower()
    else:
        contains = lambda s: needle in s

    root = root.resolve()
    for p in root.rglob("*.csv"):
        if not p.is_file():
            continue
        try:
            for lineno, line in iter_lines(p):
                if contains(line):
                    out = p
                    if relative:
                        try:
                            out = p.relative_to(root)
                        except ValueError:
                            pass
                    print(f"{out}:{lineno}")
        except (OSError, UnicodeError):
            # 无法读取的文件直接跳过
            continue

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="在目录下递归搜索CSV内容，输出包含指定字符串的 文件:行号")
    ap.add_argument("root", help="根目录")
    ap.add_argument("text", help="要搜索的字符串")
    ap.add_argument("-i", "--ignore-case", action="store_true", help="忽略大小写")
    ap.add_argument("--abs", action="store_true", help="输出绝对路径")
    args = ap.parse_args()
    search_csv(Path(args.root), args.text, ignore_case=args.ignore_case, relative=not args.abs)
