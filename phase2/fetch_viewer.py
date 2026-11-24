"""
scan_picker の立体表示に使う部品(Babylon.js)を、手元に取ってくる。

    cd phase2
    python fetch_viewer.py

phase2/vendor/ に置く(git 管理しない)。一度取ってくれば、ネットが無い所でも
scan_picker の画面が出る。版は固定してあり、同じ版が既にあれば何もしない。
"""

import sys
import urllib.request
from pathlib import Path

VERSION = "9.37.2"
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
# 置くファイル名 -> 取ってくる先
FILES = {
    "babylon.js": f"https://cdn.babylonjs.com/v{VERSION}/babylon.js",
    "babylonjs.loaders.min.js":
        f"https://cdn.babylonjs.com/v{VERSION}/loaders/babylonjs.loaders.min.js",
}
VERSION_FILE = "VERSION"


def missing():
    """足りないファイルの名前を返す(版が違うときは全部)。"""
    mark = VENDOR_DIR / VERSION_FILE
    if not mark.exists() or mark.read_text(encoding="utf-8").strip() != VERSION:
        return list(FILES)
    return [name for name in FILES if not (VENDOR_DIR / name).exists()]


def fetch(log=print):
    need = missing()
    if not need:
        log(f"Babylon.js {VERSION} は取得済みです: {VENDOR_DIR}")
        return
    VENDOR_DIR.mkdir(exist_ok=True)
    for name in need:
        log(f"取得中: {FILES[name]}")
        req = urllib.request.Request(FILES[name], headers={"User-Agent": "fetch-viewer"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        # 途中で切れたファイルを残さないよう、書き終えてから名前を付け替える
        tmp = VENDOR_DIR / (name + ".part")
        tmp.write_bytes(data)
        tmp.replace(VENDOR_DIR / name)
        log(f"  {len(data) / 1e6:.1f}MB")
    (VENDOR_DIR / VERSION_FILE).write_text(VERSION + "\n", encoding="utf-8")
    log(f"Babylon.js {VERSION} を {VENDOR_DIR} に置きました")


if __name__ == "__main__":
    try:
        fetch()
    except OSError as e:
        sys.exit(f"取得できませんでした(ネットにつながっているか確かめてください): {e}")
