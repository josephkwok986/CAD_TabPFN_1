#!/usr/bin/env python3
# 统计目录及其子目录中，后缀属于给定集合的文件数量（逐后缀与总计）

'''
python tools/count_file_by_suffix.py "/workspace/Gjj Local/data/CAD/CAD_data" step
'''
#!/usr/bin/env python3
# 更快地统计目录及子目录中指定后缀文件数量
import os
from collections import Counter
import argparse

def normalize_suffixes(suffixes, ignore_case=True):
    cleaned = []
    for s in suffixes:
        s = s.strip()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        cleaned.append(s.lower() if ignore_case else s)
    # 去重并按长度降序，避免 .gz 抢到 .tar.gz
    uniq = list(dict.fromkeys(cleaned))
    uniq.sort(key=len, reverse=True)
    return tuple(uniq)

def count_by_suffix_fast(root, suffixes, ignore_case=True, followlinks=False):
    suffs = normalize_suffixes(suffixes, ignore_case)
    counts = Counter()
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=followlinks):
        for name in filenames:
            nm = name.lower() if ignore_case else name
            for s in suffs:
                if nm.endswith(s):
                    counts[s] += 1
                    total += 1
                    break
    return counts, total

def main():
    ap = argparse.ArgumentParser(description="快速统计指定后缀文件数量")
    ap.add_argument("root", help="根目录")
    ap.add_argument("suffix", nargs="+", help="后缀列表，如 .py .csv .tar.gz 或 py csv tar.gz")
    ap.add_argument("-I", "--case-sensitive", action="store_true", help="大小写敏感，默认不敏感")
    ap.add_argument("--followlinks", action="store_true", help="跟随符号链接")
    args = ap.parse_args()

    counts, total = count_by_suffix_fast(
        args.root, args.suffix, ignore_case=not args.case_sensitive, followlinks=args.followlinks
    )
    for s in sorted(counts):
        print(f"{s}\t{counts[s]}")
    print(f"TOTAL\t{total}")

if __name__ == "__main__":
    main()
