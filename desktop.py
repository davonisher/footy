#!/usr/bin/env python3
"""Native macOS window for Footy (no browser)."""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

# Ensure project root imports work when launched from the .app
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import OUTPUT_DIR, UPLOAD_DIR, WORK_DIR, app  # noqa: E402

HOST = "127.0.0.1"
PORT = 5055
URL = f"http://{HOST}:{PORT}"


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def _wait_ready(timeout: float = 12.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.4):
                return True
        except OSError:
            time.sleep(0.15)
    return False


def _run_flask() -> None:
    # threaded=True so uploads/transcription don't block the UI API
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if _port_free(PORT):
        thread = threading.Thread(target=_run_flask, daemon=True)
        thread.start()
        if not _wait_ready():
            print("Server startte niet op tijd", file=sys.stderr)
            return 1
    else:
        # Already running — still open the native window against it
        if not _wait_ready(timeout=2.0):
            print(f"Poort {PORT} is bezet maar reageert niet", file=sys.stderr)
            return 1

    import webview

    window = webview.create_window(
        title="Footy",
        url=URL,
        width=920,
        height=640,
        min_size=(560, 420),
        background_color="#f5f5f7",
        text_select=True,
        confirm_close=False,
        easy_drag=True,
        resizable=True,
    )
    # Cocoa backend = echte macOS-webview, geen Chrome/Safari-venster
    webview.start(gui="cocoa", debug=False)
    _ = window
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
