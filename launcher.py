# =============================================================================
# launcher.py — Start RT-DETR Sentinel Pro (both servers in one command)
#
# Usage:  python launcher.py
# =============================================================================

import os
import sys
import time
import signal
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable

BANNER = r"""
  ██████╗ ████████╗      ██████╗ ███████╗████████╗██████╗
  ██╔══██╗╚══██╔══╝      ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗
  ██████╔╝   ██║   █████╗██║  ██║█████╗     ██║   ██████╔╝
  ██╔══██╗   ██║   ╚════╝██║  ██║██╔══╝     ██║   ██╔══██╗
  ██║  ██║   ██║         ██████╔╝███████╗   ██║   ██║  ██║
  ╚═╝  ╚═╝   ╚═╝         ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝
       S E N T I N E L   P R O  —  RT-DETR Object Detection Suite
"""

procs: list[subprocess.Popen] = []


def shutdown(signum=None, frame=None):
    print("\n\n🛑  Shutting down all servers…")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    # Give a moment then force-kill
    time.sleep(1.5)
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
    print("👋  Goodbye.")
    sys.exit(0)


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)


def wait_for_server(url: str, timeout: float = 20.0, label: str = "") -> bool:
    """Poll a URL until it responds or times out."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    print(f"⚠️  {label} did not respond within {timeout}s — continuing anyway.")
    return False


def main():
    print(BANNER)

    # ── 1. Stream Server (FastAPI / uvicorn) ─────────────────────────────────
    stream_cmd = [PYTHON, os.path.join(BASE_DIR, "stream_server.py")]
    print("🚀  Starting Stream Server (FastAPI)  →  http://localhost:8502")
    stream_proc = subprocess.Popen(stream_cmd, cwd=BASE_DIR)
    procs.append(stream_proc)

    # Wait for stream server to be ready
    wait_for_server("http://localhost:8502/health", timeout=20, label="Stream Server")
    print("✅  Stream Server is online")

    # ── 2. Streamlit dashboard ────────────────────────────────────────────────
    streamlit_cmd = [
        PYTHON, "-m", "streamlit", "run",
        os.path.join(BASE_DIR, "app.py"),
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]
    print("🚀  Starting Streamlit Dashboard       →  http://localhost:8501")
    streamlit_proc = subprocess.Popen(streamlit_cmd, cwd=BASE_DIR)
    procs.append(streamlit_proc)

    wait_for_server("http://localhost:8501", timeout=25, label="Streamlit")
    print("✅  Streamlit Dashboard is online")

    print("\n" + "─" * 60)
    print("  Dashboard  →  http://localhost:8501")
    print("  MJPEG feed →  http://localhost:8502/stream")
    print("  API docs   →  http://localhost:8502/docs")
    print("─" * 60)
    print("  Press  Ctrl+C  to stop all servers\n")

    # ── 3. Keep alive — restart if either crashes ─────────────────────────────
    while True:
        time.sleep(3)

        if stream_proc.poll() is not None:
            print("⚠️  Stream Server crashed — restarting…")
            stream_proc = subprocess.Popen(stream_cmd, cwd=BASE_DIR)
            procs[0] = stream_proc

        if streamlit_proc.poll() is not None:
            print("⚠️  Streamlit crashed — restarting…")
            streamlit_proc = subprocess.Popen(streamlit_cmd, cwd=BASE_DIR)
            procs[1] = streamlit_proc


if __name__ == "__main__":
    main()
