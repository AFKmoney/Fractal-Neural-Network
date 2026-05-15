"""
RemoteTrainer — SSH into any cloud GPU, launch NFN training, stream logs back.

Requires: pip install paramiko
Falls back to a clear error if paramiko is missing.
"""

import io
import json
import os
import re
import shlex
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

_METRIC_PATTERNS = {
    "step":  re.compile(r"step\s+(\d{1,12})"),
    "lm":    re.compile(r"\blm\s+([\d.]{1,16})"),
    "ppl":   re.compile(r"\bppl\s+([\d.]{1,16})"),
    "agi_w": re.compile(r"\bagi_w\s+([\d.]{1,16})"),
    "phase": re.compile(r"\[(warmup|ramp|adaptive)\]"),
    "gn":    re.compile(r"\bgn\s+([\d.]{1,16})"),
    "lr":    re.compile(r"\blr\s+([\d.eE+\-]{1,16})"),
}


def parse_metrics(line: str) -> Optional[Dict]:
    """Parse: step 100 | lm 4.21 | ppl 67.8 | agi_w 0.12 [ramp] | gn 0.65 | lr 2.8e-04"""
    m: Dict = {}
    for key, pat in _METRIC_PATTERNS.items():
        match = pat.search(line)
        if match:
            val = match.group(1)
            if key == "phase":
                m[key] = val
            else:
                try:
                    m[key] = float(val)
                except ValueError:
                    pass
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

            self.gpu_info = self._exec(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null "
                "|| echo 'No NVIDIA GPU'",
                timeout=10,
            ).split("\n")[0]
            hostname = self._exec("uname -n", timeout=10)

            return {"ok": True, "hostname": hostname, "gpu": self.gpu_info}

        except paramiko.AuthenticationException:
            return {"ok": False, "error": "Authentication failed — check username/password/key"}
        except paramiko.NoValidConnectionsError as e:
            return {"ok": False, "error": f"Cannot connect: {e}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def disconnect(self):
        self._stop_event.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)
        self._stream_thread = None
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

    def _exec(self, cmd: str, timeout: int = 60) -> str:
        """Run a remote command and return combined stdout+stderr; closes channels."""
        if not self._ssh:
            return ""
        stdin, stdout, stderr = self._ssh.exec_command(cmd, timeout=timeout)
        try:
            out = stdout.read().decode(errors="replace").strip()
            err = stderr.read().decode(errors="replace").strip()
            return out or err
        finally:
            for ch in (stdin, stdout, stderr):
                try:
                    ch.close()
                except Exception:
                    pass

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
                out = self._exec(cmd, timeout=300)
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

    @staticmethod
    def _validate_config(config: str) -> str:
        if config not in ("nano", "small", "medium", "large"):
            raise ValueError(f"config must be one of nano/small/medium/large, got: {config!r}")
        return config

    @staticmethod
    def _safe_dataset_id(name: str) -> str:
        """Allow only alphanumerics, slash, dash, underscore, dot — typical dataset IDs."""
        if not re.fullmatch(r"[A-Za-z0-9_\-./]{1,128}", name):
            raise ValueError(f"invalid dataset id: {name!r}")
        return name

    def start(self, dataset: str = "wikipedia-en-simple", hf_dataset: str = "",
              config: str = "small", batch: int = 8, seq_len: int = 512,
              lr: float = 2e-4, epochs: int = 3,
              max_chars: int = 30_000_000) -> Dict:
        if not self._ssh:
            return {"ok": False, "error": "Not connected"}

        # Validate inputs (prevents shell injection — all values land inside an SSH-exec'd shell command)
        try:
            config = self._validate_config(config)
            ds = self._safe_dataset_id(hf_dataset.strip() if hf_dataset else dataset.strip())
            batch     = max(1, min(int(batch), 1024))
            seq_len   = max(64, min(int(seq_len), 16384))
            epochs    = max(1, min(int(epochs), 1000))
            max_chars = max(1000, min(int(max_chars), 10_000_000_000))
            lr        = float(lr)
            if not (1e-7 <= lr <= 1.0):
                raise ValueError(f"lr out of range: {lr}")
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": f"Invalid argument: {e}"}

        # Stop any previous tail thread before starting a new one
        if self._stream_thread and self._stream_thread.is_alive():
            self._stop_event.set()
            self._stream_thread.join(timeout=5)
        self._stream_thread = None
        self._stop_event.clear()

        self.status = "training"
        with self._log_lock:
            self._log.clear()
        self.last_metrics = {}

        # Build training command — inputs are pre-validated, but quote them for defence in depth
        ckpt_path = f"{REMOTE_DIR}/checkpoints/agi_nfn_latest.pt"
        resume_check = (
            f"$([ -f {shlex.quote(ckpt_path)} ] "
            f"&& echo --resume && echo {shlex.quote(ckpt_path)})"
        )
        train_cmd = (
            f"cd {REMOTE_DIR} && "
            f"python cloud_train.py --dataset {shlex.quote(ds)} "
            f"--config {shlex.quote(config)} "
            f"--batch {batch} "
            f"--seq-len {seq_len} "
            f"--lr {lr:g} "
            f"--epochs {epochs} "
            f"--max-chars {max_chars} "
            f"--fp16 "
            f"--save-every 500 "
            f"{resume_check}"
        )

        # Launch inside tmux so it survives SSH drops.
        # tmux new-session passes its [shell-command] arg to /bin/sh -c, so we just
        # quote the whole pipeline as a single argument.
        inner = f"{train_cmd} 2>&1 | tee {LOG_FILE}; echo __DONE__"
        launch = (
            f"rm -f {LOG_FILE}; "
            f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; "
            f"tmux new-session -d -s {TMUX_SESSION} {shlex.quote(inner)}"
        )
        try:
            err = self._exec(launch, timeout=30)
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
            if self._exec(f"test -f {LOG_FILE} && echo yes", timeout=10) == "yes":
                break
            time.sleep(1)

        stdin, stdout, stderr = self._ssh.exec_command(f"tail -n 0 -f {LOG_FILE}")
        chan = stdout.channel
        chan.settimeout(0.5)

        try:
            buf = ""
            while not self._stop_event.is_set():
                try:
                    chunk = chan.recv(4096)
                except Exception:
                    if self._stop_event.is_set():
                        return
                    continue
                if not chunk:
                    if chan.exit_status_ready():
                        break
                    continue
                buf += chunk.decode(errors="replace")
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
        finally:
            for ch in (stdin, stdout, stderr):
                try:
                    ch.close()
                except Exception:
                    pass

    def stop(self) -> Dict:
        """Send Ctrl-C to tmux pane — training saves checkpoint then exits."""
        if not self._ssh:
            return {"ok": False, "error": "Not connected"}
        try:
            self._exec(f"tmux send-keys -t {TMUX_SESSION} C-c ENTER", timeout=10)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        # Wait briefly for the trainer to flush its checkpoint, then signal the tail thread
        time.sleep(3)
        self._stop_event.set()
        self.status = "done"
        return {"ok": True}

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
