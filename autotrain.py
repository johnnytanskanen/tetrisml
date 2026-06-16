"""
Background training helper, shared by server.py and tetris_ai.py.

When you launch either front-end, this spins up `train.py` in the background so
the AI keeps learning (writing weights.json) while you watch. The game hot-reloads
those weights, so you can see it improve in real time.

Design choices:
  * uses only half your CPU cores, so the UI stays responsive
  * a lockfile prevents two launchers from spawning two trainers
  * the trainer is stopped when the launcher exits (no lingering CPU)
  * set TETRIS_NO_TRAIN=1 to disable entirely
"""
import atexit
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, '.train.lock')
LOG = os.path.join(HERE, 'training.log')


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _trainer_running():
    try:
        with open(LOCK) as f:
            pid = int(f.read().strip())
        return _alive(pid)
    except (OSError, ValueError):
        return False


def start_background_training():
    """Spawn train.py in the background. Returns the Popen, or None if skipped."""
    if os.environ.get('TETRIS_NO_TRAIN'):
        return None
    if _trainer_running():
        return None  # another launcher already has one going

    workers = max(1, (os.cpu_count() or 2) // 2)
    args = [sys.executable, os.path.join(HERE, 'train.py'),
            '--resume', '--workers', str(workers),
            '--generations', '40', '--population', '30', '--pieces', '300']
    try:
        log = open(LOG, 'a')
        # new session/group so we can reliably kill train.py *and* its Pool workers
        proc = subprocess.Popen(args, stdout=log, stderr=log, cwd=HERE,
                                start_new_session=True)
    except OSError:
        return None

    with open(LOCK, 'w') as f:
        f.write(str(proc.pid))
    atexit.register(lambda: stop_background_training(proc))
    return proc


def stop_background_training(proc):
    if proc is None:
        return
    import signal
    try:
        # kill the whole group (train.py + its multiprocessing workers)
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            proc.terminate()
        except (OSError, ProcessLookupError):
            pass
    try:
        proc.wait(timeout=3)  # reap so it doesn't linger as a zombie
    except Exception:
        pass
    try:
        with open(LOCK) as f:
            if f.read().strip() == str(proc.pid):
                os.remove(LOCK)
    except (OSError, ValueError):
        pass
