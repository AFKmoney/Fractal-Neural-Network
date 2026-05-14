"""
RemoteTrainer — SSH into any cloud GPU, launch NFN training, stream logs back.

Requires: pip install paramiko
Falls back to a clear error if paramiko is missing.
"""

import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

try:
    import paramiko
    _HAS_PARAMIKO = True
except ImportError:
    _HAS_PARAMIKO = False

try:
    import urllib.request as _urllib
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

REPO_URL    = "https://github.com/AFKmoney/FNN.git"
REMOTE_DIR  = "~/FNN"
TMUX_SESSION = "nfn_train"
LOG_FILE    = f"{REMOTE_DIR}/train_remote.log"


# ── Metric line parser ─────────────────────────────────────────────────────────

def parse_metrics(line: str) -> Optional[Dict]:
    """Parse: step 100 | lm 4.21 | ppl 67.8 | agi_w 0.12 [ramp] | gn 0.65 | lr 2.8e-04"""
    patterns = {
        "step":  r"step\s+(\d+)",
        "lm":    r"\blm\s+([\d.]+)",
        "ppl":   r"\bppl\s+([\d.]+)",
        "agi_w": r"\bagi_w\s+([\d.]+)",
        "phase": r"\[(warmup|ramp|adaptive)\]",
        "gn":    r"\bgn\s+([\d.]+)",
        "lr":    r"\blr\s+([\de.+\-]+)",
    }
    m: Dict = {}
    for key, pat in patterns.items():
        match = re.search(pat, line)
        if match:
            val = match.group(1)
            m[key] = val if key == "phase" else float(val)
    return m if "step" in m else None


# ── RemoteTrainer ──────────────────────────────────────────────────────────────

class RemoteTrainer:
    """Manages an SSH connection to a cloud GPU and runs NFN training."""

    def __init__(self):
        self._ssh: Optional["paramiko.SSHClient"] = None
        self._sftp = None
        self.status: str = "disconnected"
        # Thread-safe log accumulator
        self._log: List[str] = []
        self._log_lock = threading.Lock()
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.last_metrics: Dict = {}
        self.gpu_info: str = "unknown"

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self, host: str, user: str, port: int = 22,
                password: str = "", key_data: str = "") -> Dict:
        if not _HAS_PARAMIKO:
            return {"ok": False,
                    "error": "paramiko not installed on server — run: pip install paramiko"}

        self.disconnect()
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            kw: Dict = dict(hostname=host, port=port, username=user, timeout=20)
            if key_data.strip():
                key_data = key_data.strip()
                kf = io.StringIO(key_data)
                pkey = None
                for cls in (paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey,
                            paramiko.DSSKey):
                    try:
                        kf.seek(0)
                        pkey = cls.from_private_key(kf)
                        break
                    except Exception:
                        continue
                if pkey is None:
                    return {"ok": False, "error": "Could not parse SSH private key (RSA/Ed25519/ECDSA/DSS)"}
                kw["pkey"] = pkey
            elif password:
                kw["password"] = password
            else:
                return {"ok": False, "error": "Provide SSH key or password"}

            client.connect(**kw)
            self._ssh = client
            self._sftp = client.open_sftp()
            self.status = "connected"

            # Gather GPU info
            _, out, _ = client.exec_command(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No NVIDIA GPU'"
            )
            self.gpu_info = out.read().decode().strip().split("\n")[0]

            _, out2, _ = client.exec_command("uname -n")
            hostname = out2.read().decode().strip()

            return {"ok": True, "hostname": hostname, "gpu": self.gpu_info}

        except paramiko.AuthenticationException:
            return {"ok": False, "error": "Authentication failed — check username/password/key"}
        except paramiko.NoValidConnectionsError as e:
            return {"ok": False, "error": f"Cannot connect: {e}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def disconnect(self):
        self._stop_event.set()
        for obj in (self._sftp, self._ssh):
            if obj:
                try:
                    obj.close()
                except Exception:
                    pass
        self._ssh = None
        self._sftp = None
        self.status = "disconnected"
        self._stop_event.clear()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def setup(self, progress_cb: Callable[[str], None] = None) -> Dict:
        """Clone repo + install deps on remote machine."""
        if not self._ssh:
            return {"ok": False, "error": "Not connected"}
        self.status = "setting_up"
        try:
            steps = [
                ("Checking Python…",
                 "python3 --version 2>&1"),
                ("Cloning / updating repo…",
                 f"if [ -d {REMOTE_DIR} ]; then git -C {REMOTE_DIR} pull --ff-only 2>&1; "
                 f"else git clone {REPO_URL} {REMOTE_DIR} 2>&1; fi"),
                ("Installing dependencies…",
                 f"cd {REMOTE_DIR} && pip install -e . -q 2>&1 | tail -3"),
                ("Installing extras…",
                 "pip install tqdm paramiko -q 2>&1 | tail -2"),
            ]
            for msg, cmd in steps:
                if progress_cb:
                    progress_cb(msg)
                _, stdout, stderr = self._ssh.exec_command(cmd, timeout=300)
                out = (stdout.read().decode().strip() or stderr.read().decode().strip())
                if progress_cb and out:
                    progress_cb(out[:300])

            self.status = "connected"
            return {"ok": True}
        except Exception as e:
            self.status = "error"
            return {"ok": False, "error": str(e)}

    # ── HuggingFace dataset search ─────────────────────────────────────────────

    @staticmethod
    def search_hf(query: str, limit: int = 8) -> List[Dict]:
        if not _HAS_URLLIB or not query.strip():
            return []
        try:
            url = (
                f"https://huggingface.co/api/datasets"
                f"?search={query.strip()}&limit={limit}&sort=downloads&direction=-1"
            )
            req = _urllib.Request(url, headers={"User-Agent": "NFN/5.1"})
            with _urllib.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read())
            return [
                {
                    "id":          d.get("id", ""),
                    "downloads":   d.get("downloads", 0),
                    "description": (d.get("cardData", {}) or {}).get("pretty_name",
                                   (d.get("description") or ""))[:100],
                }
                for d in data
                if d.get("id")
            ]
        except Exception:
            return []

    # ── Training ───────────────────────────────────────────────────────────────

    def start(self, dataset: str = "wikipedia-en-simple", hf_dataset: str = "",
              config: str = "small", batch: int = 8, seq_len: int = 512,
              lr: float = 2e-4, epochs: int = 3,
              max_chars: int = 30_000_000) -> Dict:
        if not self._ssh:
            return {"ok": False, "error": "Not connected"}

        self.status = "training"
        self._stop_event.clear()
        with self._log_lock:
            self._log.clear()
        self.last_metrics = {}

        # Build training command
        dataset_arg = f"--dataset {hf_dataset}" if hf_dataset else f"--dataset {dataset}"
        resume_check = (
            f"$([ -f {REMOTE_DIR}/checkpoints/agi_nfn_latest.pt ] "
            f"&& echo '--resume {REMOTE_DIR}/checkpoints/agi_nfn_latest.pt')"
        )
        train_cmd = (
            f"cd {REMOTE_DIR} && "
            f"python cloud_train.py {dataset_arg} "
            f"--config {config} "
            f"--batch {batch} "
            f"--seq-len {seq_len} "
            f"--lr {lr} "
            f"--epochs {epochs} "
            f"--max-chars {max_chars} "
            f"--fp16 "
            f"--save-every 500 "
            f"{resume_check}"
        )

        # Clear old log and launch in tmux
        launch = (
            f"rm -f {LOG_FILE}; "
            f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; "
            f"tmux new-session -d -s {TMUX_SESSION} "
            f"\"bash -c '{train_cmd} 2>&1 | tee {LOG_FILE}; echo __DONE__'\""
        )
        try:
            _, _, stderr = self._ssh.exec_command(launch)
            err = stderr.read().decode().strip()
            # tmux kill-session prints to stderr when session doesn't exist — ignore it
            bad = [l for l in err.splitlines() if "kill-session" not in l and l.strip()]
            if bad:
                self.status = "error"
                return {"ok": False, "error": "\n".join(bad)}

            self._stream_thread = threading.Thread(target=self._tail_log, daemon=True)
            self._stream_thread.start()
            return {"ok": True}
        except Exception as e:
            self.status = "error"
            return {"ok": False, "error": str(e)}

    def _tail_log(self):
        """Background thread: SSH tail -f the remote log file."""
        if not self._ssh:
            return
        # Wait up to 30 s for log file to appear
        for _ in range(30):
            if self._stop_event.is_set():
                return
            _, out, _ = self._ssh.exec_command(f"test -f {LOG_FILE} && echo yes")
            if out.read().decode().strip() == "yes":
                break
            time.sleep(1)

        _, stdout, _ = self._ssh.exec_command(f"tail -n 0 -f {LOG_FILE}")
        stdout.channel.setblocking(False)

        try:
            buf = ""
            while not self._stop_event.is_set():
                try:
                    chunk = stdout.read(4096)
                    if not chunk:
                        time.sleep(0.2)
                        continue
                    buf += chunk.decode(errors="replace")
                except Exception:
                    time.sleep(0.2)
                    continue

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.rstrip()
                    with self._log_lock:
                        self._log.append(line)
                    m = parse_metrics(line)
                    if m:
                        self.last_metrics = m
                    if "__DONE__" in line:
                        self.status = "done"
                        return
        except Exception as e:
            with self._log_lock:
                self._log.append(f"[log stream error] {e}")

    def stop(self) -> Dict:
        """Send Ctrl-C to tmux pane — training saves checkpoint then exits."""
        if not self._ssh:
            return {"ok": False, "error": "Not connected"}
        self._stop_event.set()
        try:
            self._ssh.exec_command(f"tmux send-keys -t {TMUX_SESSION} C-c ENTER")
            time.sleep(3)
            self.status = "done"
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Checkpoints ────────────────────────────────────────────────────────────

    def list_checkpoints(self) -> List[Dict]:
        if not self._sftp:
            return []
        try:
            remote_ckpt = f"{REMOTE_DIR}/checkpoints"
            files = self._sftp.listdir_attr(remote_ckpt)
            return sorted(
                [{"name": f.filename, "size_mb": round(f.st_size / 1e6, 1)} for f in files],
                key=lambda x: x["name"],
            )
        except Exception:
            return []

    def download_checkpoint(self, name: str, dest: str) -> Dict:
        if not self._sftp:
            return {"ok": False, "error": "Not connected"}
        try:
            remote_path = f"{REMOTE_DIR}/checkpoints/{name}"
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            self._sftp.get(remote_path, dest)
            size = os.path.getsize(dest)
            return {"ok": True, "path": dest, "size_mb": round(size / 1e6, 1)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Log access (thread-safe snapshot) ─────────────────────────────────────

    def get_logs(self, since: int = 0) -> List[str]:
        with self._log_lock:
            return list(self._log[since:])
