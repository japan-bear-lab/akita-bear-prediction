"""web/ を簡易 HTTP サーバで配信する。

スマホ（同じ Wi-Fi）から Mac の LAN IP で地図を開けるようにするためのもの。
`open web/index.html`（file://）だとスマホから到達できないため。

使い方:
    python serve.py            # ポート 8000
    python serve.py 8080       # ポート指定

実行すると、スマホで開く用の URL（http://<MacのLAN-IP>:8000/）が表示される。
Mac とスマホが同じ Wi-Fi にいる必要がある。停止は Ctrl+C。
"""

from __future__ import annotations

import socket
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_PORT = 8000


def lan_ip() -> str:
    """LAN 内で他端末から到達できる自分の IP を推定する。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # 実際には送信しない。経路上の自 IP を得るだけ
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        # points.js を更新したらすぐ反映されるようキャッシュ無効化
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):  # ログを静かに
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    if not (WEB_DIR / "index.html").exists():
        print(f"⚠ {WEB_DIR}/index.html が見つかりません。")
        sys.exit(1)
    if not (WEB_DIR / "points.js").exists():
        print("⚠ web/points.js が未生成です。先に `python -m src.build_db` を実行してください。")

    ip = lan_ip()
    print("\n🐻 クマ出没マップ サーバ起動")
    print(f"  📱 スマホ（同じ Wi-Fi）: http://{ip}:{port}/")
    print(f"  💻 この Mac:              http://localhost:{port}/")
    print("  停止: Ctrl+C\n")

    with ThreadingHTTPServer(("0.0.0.0", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n停止しました。")


if __name__ == "__main__":
    main()
