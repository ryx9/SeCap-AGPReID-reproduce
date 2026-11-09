"""Start the lightweight 45-minute Kaggle watcher as a detached local process."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
state = root / 'kaggle/monitor'
state.mkdir(exist_ok=True)
pidfile = state / 'monitor.pid'
if pidfile.exists():
    pid = int(pidfile.read_text())
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'kaggle/watch.py' in args and b'--loop' in args:
            print(f'Monitor already running: PID {pid}')
            sys.exit(0)
    except FileNotFoundError:
        pass
with (state / 'controller.log').open('a') as log:
    process = subprocess.Popen([sys.executable, '-u', 'kaggle/watch.py', '--loop', '--resume-sessions'],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True)
pidfile.write_text(str(process.pid))
print(f'Started Kaggle monitor PID {process.pid}; logs: {state / "controller.log"}')
