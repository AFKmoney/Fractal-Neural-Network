#!/usr/bin/env python3
"""
NFN AGI — All-in-one launcher
==============================
Usage:
    python run.py                          # nano model, port 8000, open browser
    python run.py --config small           # small model preset
    python run.py --model checkpoints/agi_nfn_final.pt
    python run.py --port 8080 --no-open
    python run.py --ttl                    # enable test-time learning at startup
    python run.py --host 0.0.0.0           # expose to LAN

Options:
    --host HOST          Bind host (default: 127.0.0.1)
    --port PORT          TCP port  (default: 8000)
    --model PATH         Pre-load checkpoint (.pt file)
    --config NAME        Model size preset: nano | small | medium
    --open / --no-open   Auto-open browser tab (default: --open)
    --ttl                Enable test-time adaptation at startup
    --adapter-rank N     LoRA adapter rank for TTL (default: 8)
    --reload             Dev mode with auto-reload
"""

import argparse
import json
import os
import sys
import webbrowser
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

BANNER = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║   ⬡  Neural Fractal Network — AGI All-in-One App  ⬡         ║
  ║                                                              ║
  ║   • Chat with the model          /ws/chat                    ║
  ║   • Live training                /api/train/start            ║
  ║   • Web exploration (auto-learn) /ws/explore                 ║
  ║   • Test-time adaptation (TTL)   /api/ttl/enable             ║
  ╚══════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NFN AGI — Web interface + autonomous learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host",          type=str,   default="127.0.0.1",
                   help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port",          type=int,   default=8000,
                   help="TCP port (default: 8000)")
    p.add_argument("--model",         type=str,   default=None,
                   help="Path to a .pt checkpoint to preload")
    p.add_argument("--config",        type=str,   default="nano",
                   choices=["nano", "small", "medium"],
                   help="Model size preset (default: nano)")
    p.add_argument("--open",          dest="open_browser",
                   action="store_true",  default=True,
                   help="Open browser automatically (default)")
    p.add_argument("--no-open",       dest="open_browser",
                   action="store_false",
                   help="Do not open the browser")
    p.add_argument("--ttl",           action="store_true", default=False,
                   help="Enable test-time adaptation (TTL) at startup")
    p.add_argument("--adapter-rank",  type=int,   default=8,
                   help="LoRA adapter rank for TTL (default: 8)")
    p.add_argument("--reload",        action="store_true", default=False,
                   help="Dev mode with auto-reload")
    return p.parse_args()


def _open_browser(url: str, delay: float = 1.5) -> None:
    """Open browser tab after a short delay (so the server has time to start)."""
    def _do_open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_do_open, daemon=True).start()


def _try_pywebview(url: str, title: str = "NFN AGI") -> bool:
    """
    Try to open a native pywebview window.
    Returns True if successful, False if pywebview is not installed.
    """
    try:
        import webview  # type: ignore[import]
        webview.create_window(title, url, width=1280, height=800, resizable=True)
        webview.start()
        return True
    except ImportError:
        return False
    except Exception:
        return False


def main() -> None:
    args = parse_args()

    url = f"http://{args.host}:{args.port}"

    print(BANNER)
    print(f"  URL        : {url}")
    print(f"  Config     : {args.config}")
    print(f"  Model      : {args.model or 'auto (nano)'}")
    print(f"  TTL        : {'enabled (rank=' + str(args.adapter_rank) + ')' if args.ttl else 'disabled'}")
    print(f"  Dev reload : {'yes' if args.reload else 'no'}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    # Write runtime config so interface/app.py can pick it up
    runtime = {
        "config_name":   args.config,
        "model_path":    args.model,
        "ttl":           args.ttl,
        "adapter_rank":  args.adapter_rank,
    }
    runtime_path = ROOT / ".nfn_runtime.json"
    runtime_path.write_text(json.dumps(runtime))

    # Import uvicorn early so we get a clear error message
    try:
        import uvicorn
    except ImportError:
        print("  [ERROR] uvicorn not installed.")
        print("  Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)

    # Decide how to open the UI
    use_pywebview = False
    if args.open_browser:
        # Try native window first; fall back to browser tab
        # We can only try pywebview *after* the server is running,
        # so we just schedule the browser open for now.
        _open_browser(url, delay=1.5)

    try:
        uvicorn.run(
            "interface.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n  Shutting down. Goodbye!")
    except OSError as e:
        if "10048" in str(e) or "address already in use" in str(e).lower():
            print(f"\n  [ERROR] Port {args.port} is already in use.")
            print(f"  Another instance may still be running.")
            print(f"  Fix:  python run.py --port {args.port + 1}")
            print(f"  Or close the other instance first, then retry.")
            sys.exit(1)
        raise
    finally:
        if runtime_path.exists():
            try:
                runtime_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
