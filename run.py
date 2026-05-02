#!/usr/bin/env python3
"""
NFN Interface Launcher

Usage:
    python run.py                         # start on default port 8000
    python run.py --port 8080
    python run.py --model checkpoints/nfn_final.pt
    python run.py --config small          # use small preset
"""

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="NFN Web Interface")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", type=str, default=None, help="Checkpoint path to pre-load")
    p.add_argument("--config", type=str, default="nano",
                   choices=["nano", "small", "medium"])
    p.add_argument("--open", action="store_true", default=True,
                   help="Open browser automatically")
    p.add_argument("--no-open", dest="open", action="store_false")
    p.add_argument("--reload", action="store_true", help="Dev mode with auto-reload")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║       Neural Fractal Network — Interface v1.0            ║
║  Réseau neuronal à topologie fractale et couplage        ║
║  sinusoïdal paramétrique — Philippe-Antoine Robert       ║
╚══════════════════════════════════════════════════════════╝

  → URL: http://{args.host}:{args.port}
  → Config: {args.config}
  → Model: {args.model or 'auto-detect'}

  Tabs: Chat · Code · Agent · Training · Model Info
  Press Ctrl+C to stop.
""")

    # Write runtime config for the app to pick up
    runtime = {
        "config_name": args.config,
        "model_path": args.model,
    }
    runtime_path = ROOT / ".nfn_runtime.json"
    runtime_path.write_text(json.dumps(runtime))

    url = f"http://{args.host}:{args.port}"
    if args.open:
        import threading
        def _open():
            import time; time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        import uvicorn
        uvicorn.run(
            "interface.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)
    finally:
        if runtime_path.exists():
            runtime_path.unlink()


if __name__ == "__main__":
    main()
