"""List every image/video the deck expects, in slide order, and whether it exists.

    python scripts/assets.py            # all assets
    python scripts/assets.py --missing  # only the ones still needed
"""
import re
import sys
from html import unescape
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parent.parent
html = (root / "index.html").read_text(encoding="utf8")
html = re.sub(r"<!--.*?-->", "", html, flags=re.S)   # skip commented-out slides
missing_only = "--missing" in sys.argv

seen = set()
slide = 0
for line in html.splitlines():
    if "<section" in line:
        slide += 1
    m = re.search(r'data-file="([^"]+)"(?:[^>]*data-alt="([^"]*)")?', line)
    if not m or m.group(1) in seen:
        continue
    seen.add(m.group(1))
    ok = (root / m.group(1)).exists()
    if missing_only and ok:
        continue
    print(f"{'ok ' if ok else '-- '} #{slide:>2}  {m.group(1):<34} {unescape(m.group(2) or '')}")
