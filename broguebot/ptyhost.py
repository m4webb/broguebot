"""Run brogue on a pty we own: no tmux, no subprocesses, no screen polling.

The patched binary writes its ready sentinel to $BROGUE_READY_FILE; here
that path is a FIFO, so "wait until the game wants input" is a blocking
select() on the read end, and the screen state is read straight from the
pty master fd into an in-process terminal emulator (vterm.VTerm). The
sentinel is emitted after the final frame is flushed, so once it arrives
a single drain of the pty yields the complete, settled screen.
"""

import fcntl
import os
import select
import shlex
import signal
import struct
import subprocess
import termios
import time

from .vterm import VTerm


KEYMAP = {
    "Enter": b"\r",
    "Escape": b"\x1b",
    "Space": b" ",
    "Tab": b"\t",
    "Up": b"\x1b[A", "Down": b"\x1b[B", "Right": b"\x1b[C", "Left": b"\x1b[D",
}


def key_bytes(key: str, literal: bool) -> bytes:
    if literal:
        return key.encode()
    if key in KEYMAP:
        return KEYMAP[key]
    if len(key) == 3 and key.startswith("C-"):
        return bytes([ord(key[2].lower()) & 0x1F])
    return key.encode()


class FifoSync:
    """ReadySync over a FIFO: a blocking read instead of stat-polling."""

    def __init__(self, pane: "PtyPane"):
        self.pane = pane

    def reset(self):
        self.pane.drain_ready()
        self.pane.ready_count = 0

    def count(self) -> int:
        self.pane.drain_ready()
        return self.pane.ready_count

    def wait(self, target: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            self.pane.drain_ready()
            if self.pane.ready_count >= target:
                self.pane.drain_output()
                return True
            remain = deadline - time.monotonic()
            if remain <= 0:
                return False
            r, _, _ = select.select(
                [fd for fd in (self.pane.ready_fd, self.pane.master)
                 if fd is not None], [], [], remain)
            if self.pane.master in r:
                self.pane.drain_output()


class PtyPane:
    """Drop-in replacement for tmux.Pane backed by a private pty."""

    def __init__(self, cols: int = 100, rows: int = 34):
        self.cols, self.rows = cols, rows
        self.vt = VTerm(cols, rows)
        self.proc: subprocess.Popen | None = None
        self.master: int | None = None
        self.ready_fd: int | None = None
        self.ready_path: str | None = None
        self.ready_count = 0

    def make_sync(self, path: str) -> FifoSync:
        self._open_ready(path)
        return FifoSync(self)

    # ------------------------------------------------------------ lifecycle

    def _open_ready(self, path: str):
        if self.ready_path == path and self.ready_fd is not None:
            return
        if self.ready_fd is not None:
            os.close(self.ready_fd)
        import stat as _st
        if os.path.exists(path):
            if not _st.S_ISFIFO(os.stat(path).st_mode):
                os.unlink(path)  # leftover regular flag file from tmux mode
                os.mkfifo(path)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.mkfifo(path)
        # open read end non-blocking so we don't wait for a writer
        self.ready_fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.ready_path = path
        self.ready_count = 0

    def respawn(self, command: str, cwd: str):
        """Start (or restart) the command. Mirrors tmux respawn-pane -k."""
        self.kill()
        argv = shlex.split(command)
        env = os.environ.copy()
        if argv and argv[0] == "exec":
            argv = argv[1:]
        if argv and argv[0] == "env":
            argv = argv[1:]
            while argv and "=" in argv[0]:
                k, v = argv[0].split("=", 1)
                env[k] = v
                argv = argv[1:]
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"  # -> brogue's 24-bit renderer
        env["LINES"] = str(self.rows)
        env["COLUMNS"] = str(self.cols)
        if "BROGUE_READY_FILE" in env:
            self._open_ready(env["BROGUE_READY_FILE"])
        master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ,
                    struct.pack("HHHH", self.rows, self.cols, 0, 0))
        self.proc = subprocess.Popen(
            argv, stdin=slave, stdout=slave, stderr=slave,
            cwd=cwd, env=env, start_new_session=True, close_fds=True)
        os.close(slave)
        os.set_blocking(master, False)
        self.master = master
        self.vt = VTerm(self.cols, self.rows)

    def kill(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        if self.master is not None:
            os.close(self.master)
            self.master = None

    def exists(self) -> bool:
        return True

    def is_dead(self) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return True
        return False

    # ------------------------------------------------------------ io

    def drain_output(self):
        """Feed everything currently buffered on the pty to the emulator."""
        if self.master is None:
            return
        while True:
            try:
                data = os.read(self.master, 65536)
            except BlockingIOError:
                return
            except OSError:  # EIO: child exited and slave closed
                return
            if not data:
                return
            self.vt.feed(data)

    def drain_ready(self):
        if self.ready_fd is None:
            return
        while True:
            try:
                data = os.read(self.ready_fd, 4096)
            except BlockingIOError:
                return
            if not data:
                return
            self.ready_count += len(data)

    def send(self, *keys: str, literal: bool = False):
        if self.master is None:
            return
        data = b"".join(key_bytes(k, literal) for k in keys)
        os.write(self.master, data)

    # ------------------------------------------------------------ capture

    def capture(self) -> list[str]:
        self.drain_output()
        return self.vt.lines()

    def capture_colors(self, start: int, end: int) -> list[str]:
        self.drain_output()
        return self.vt.sgr_lines(max(start, 0), end)

    def capture_stable(self, interval: float = 0.02, timeout: float = 4.0,
                       settle: int = 2) -> list[str]:
        """Fallback for paths with no sentinel: wait until output goes quiet."""
        self.drain_output()
        last = self.vt.lines()
        stable = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.master is None:
                return last
            r, _, _ = select.select([self.master], [], [], interval)
            if r:
                self.drain_output()
                cur = self.vt.lines()
                if cur != last:
                    stable = 0
                    last = cur
                    continue
            stable += 1
            if stable >= settle:
                return last
        return last
