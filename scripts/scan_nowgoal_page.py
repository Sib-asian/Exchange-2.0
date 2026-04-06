"""Scan a Nowgoal H2H/Live URL for extractable JS variables (one-off / dev)."""
from __future__ import annotations

import json
import re
import sys
import urllib.request


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://live5.nowgoal26.com/match/h2h-2788294"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
    print("URL:", url)
    print("HTML length:", len(html))

    vs_names = sorted(set(re.findall(r"var\s+(Vs_[a-zA-Z0-9_]+)\s*=", html)))
    print("Vs_* variables:", vs_names)

    for vn in vs_names:
        m = re.search(rf"{re.escape(vn)}\s*=\s*(\[[\s\S]+?\]\s*\])\s*;", html)
        if not m:
            continue
        s = m.group(1)
        s2 = re.sub(r"'([^']*)'", r'"\1"', s)
        try:
            data = json.loads(s2)
            lens = [len(r) for r in data[:8]]
            print(f"  {vn}: n_rows={len(data)}, row_lengths(head)={lens}")
        except Exception as e:
            print(f"  {vn}: JSON error {e}")


if __name__ == "__main__":
    main()
