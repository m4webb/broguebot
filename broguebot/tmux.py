"""Thin wrapper around the tmux CLI for driving and reading the Brogue pane."""

import os
import subprocess
import time

BROGUE_BIN = "/opt/brogue-ce/brogue"
# bot-built binary with the ready-sentinel patch (see vendor/); the user's
# interactive brogue install stays stock
BOT_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "vendor", "BrogueCE-1.15.1", "bin", "brogue")


class ReadySync:
    """Synchronize on the patched binary's ready-sentinel file.

    The patched brogue appends one byte every time it starts waiting for a
    player command, giving an exact input->render->ready handshake instead
    of screen-stability polling.
    """

    def __init__(self, path: str):
        self.path = path

    def reset(self):
        open(self.path, "w").close()

    def count(self) -> int:
        try:
            return os.stat(self.path).st_size
        except FileNotFoundError:
            return 0

    def wait(self, target: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.count() >= target:
                # tiny margin so tmux finishes parsing brogue's last draw
                time.sleep(0.012)
                return True
            time.sleep(0.004)
        return False


def run(*args: str) -> str:
    """Run a tmux command and return stdout (stripped)."""
    res = subprocess.run(["tmux", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise TmuxError(f"tmux {' '.join(args)}: {res.stderr.strip()}")
    return res.stdout.rstrip("\n")


class TmuxError(RuntimeError):
    pass


class Pane:
    """A handle to a single tmux pane running Brogue."""

    def __init__(self, target: str):
        self.target = target

    def make_sync(self, path: str) -> "ReadySync":
        return ReadySync(path)

    def exists(self) -> bool:
        try:
            run("display", "-p", "-t", self.target, "#{pane_id}")
            return True
        except TmuxError:
            return False

    def is_dead(self) -> bool:
        """True if the pane's command has exited (remain-on-exit keeps the pane)."""
        try:
            return run("display", "-p", "-t", self.target, "#{pane_dead}") == "1"
        except TmuxError:
            return True

    def size(self) -> tuple[int, int]:
        out = run("display", "-p", "-t", self.target, "#{pane_width} #{pane_height}")
        w, h = out.split()
        return int(w), int(h)

    def capture_colors(self, start: int, end: int) -> list[str]:
        """Capture rows [start..end] with SGR color escapes preserved."""
        out = subprocess.run(
            ["tmux", "capture-pane", "-p", "-e", "-t", self.target,
             "-S", str(start), "-E", str(end)],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            raise TmuxError(out.stderr.strip())
        return out.stdout.split("\n")

    def capture(self) -> list[str]:
        out = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", self.target],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            raise TmuxError(out.stderr.strip())
        return out.stdout.split("\n")

    def capture_stable(self, interval: float = 0.04, timeout: float = 4.0,
                       settle: int = 2) -> list[str]:
        """Capture repeatedly until the screen stops changing.

        Returns the last capture even if the timeout is hit (animations,
        auto-explore, resting all eventually pause for input).
        """
        last = self.capture()
        stable = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(interval)
            cur = self.capture()
            if cur == last:
                stable += 1
                if stable >= settle:
                    return cur
            else:
                stable = 0
                last = cur
        return last

    def send(self, *keys: str, literal: bool = False) -> None:
        """Send keys to the pane.

        With literal=True every argument is sent as literal text. Otherwise
        tmux key names (Escape, Space, Enter, C-x, ...) are interpreted.
        """
        args = ["send-keys", "-t", self.target]
        if literal:
            args.append("-l")
        run(*args, *keys)

    def respawn(self, command: str, cwd: str) -> None:
        """Kill whatever runs in the pane and start a fresh command."""
        run("respawn-pane", "-k", "-t", self.target, "-c", cwd, command)


def brogue_command(seed: int | None = None, wizard: bool = False,
                   ready_file: str | None = None) -> str:
    """Shell command that starts a new terminal-mode game, skipping the menu."""
    use_bot_bin = ready_file and os.path.exists(BOT_BIN)
    binary = BOT_BIN if use_bot_bin else BROGUE_BIN
    envprefix = f"env BROGUE_READY_FILE='{ready_file}' " if use_bot_bin else ""
    cmd = f"exec {envprefix}{binary} -t -E -n"
    if wizard:
        cmd += " -W"
    if seed is not None:
        cmd += f" -s {seed}"
    return cmd
