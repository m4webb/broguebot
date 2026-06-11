"""Minimal in-process terminal emulator for Brogue's escape vocabulary.

Brogue's truecolor renderer (term.c, COLORTERM=truecolor) emits only:
  \\033[48;2;R;G;Bm  background color
  \\033[38;2;R;G;Bm  foreground color
  \\033[ROW;COLf      cursor position
plus window-title/resize escapes and whatever ncurses produces during
startup (alt-screen, clear, cursor moves). This implements that subset —
enough to mirror the screen exactly without tmux in the loop.
"""

BLACK = (0, 0, 0)


class VTerm:
    def __init__(self, cols: int = 100, rows: int = 34):
        self.cols = cols
        self.rows = rows
        self.chars = [[" "] * cols for _ in range(rows)]
        self.bgs = [[BLACK] * cols for _ in range(rows)]
        self.cur_r = 0
        self.cur_c = 0
        self.bg = BLACK
        self._tail = b""  # partial escape sequence carried across feeds

    # ------------------------------------------------------------ output

    def lines(self) -> list[str]:
        """Plain text screen, like `tmux capture-pane -p`."""
        return ["".join(row).rstrip() for row in self.chars]

    def sgr_lines(self, start: int, end: int) -> list[str]:
        """Rows [start..end] with background SGR codes, like capture -e."""
        out = []
        for r in range(max(start, 0), min(end, self.rows - 1) + 1):
            parts = []
            cur = BLACK
            for c in range(self.cols):
                bg = self.bgs[r][c]
                if bg != cur:
                    parts.append("\x1b[48;2;%d;%d;%dm" % bg)
                    cur = bg
                parts.append(self.chars[r][c])
            out.append("".join(parts))
        return out

    # ------------------------------------------------------------ input

    def feed(self, data: bytes):
        data = self._tail + data
        self._tail = b""
        i, n = 0, len(data)
        while i < n:
            b = data[i]
            if b == 0x1B:  # ESC
                j = self._escape(data, i)
                if j < 0:  # incomplete sequence: stash and wait for more
                    self._tail = data[i:]
                    return
                i = j
            elif b == 0x0D:  # CR
                self.cur_c = 0
                i += 1
            elif b == 0x0A:  # LF
                self.cur_r = min(self.cur_r + 1, self.rows - 1)
                i += 1
            elif b == 0x08:  # BS
                self.cur_c = max(self.cur_c - 1, 0)
                i += 1
            elif b == 0x09:  # TAB
                self.cur_c = min((self.cur_c // 8 + 1) * 8, self.cols - 1)
                i += 1
            elif 0x20 <= b < 0x7F:
                if self.cur_c >= self.cols:
                    self.cur_c = 0
                    self.cur_r = min(self.cur_r + 1, self.rows - 1)
                self.chars[self.cur_r][self.cur_c] = chr(b)
                self.bgs[self.cur_r][self.cur_c] = self.bg
                self.cur_c += 1
                i += 1
            else:  # other control bytes: ignore
                i += 1

    def _escape(self, data: bytes, i: int) -> int:
        """Handle the escape sequence at data[i]. Returns the index after
        it, or -1 if the sequence is incomplete."""
        n = len(data)
        if i + 1 >= n:
            return -1
        kind = data[i + 1]
        if kind == ord("["):  # CSI
            j = i + 2
            while j < n and not (0x40 <= data[j] <= 0x7E):
                j += 1
            if j >= n:
                return -1
            self._csi(data[i + 2:j].decode("ascii", "replace"), chr(data[j]))
            return j + 1
        if kind == ord("]"):  # OSC ... BEL or ESC backslash
            j = i + 2
            while j < n:
                if data[j] == 0x07:
                    return j + 1
                if data[j] == 0x1B and j + 1 < n and data[j + 1] == ord("\\"):
                    return j + 2
                j += 1
            return -1
        if kind in b"()#":  # charset designation: ESC ( X
            return i + 3 if i + 2 < n else -1
        if kind == ord("M"):  # reverse index
            self.cur_r = max(self.cur_r - 1, 0)
            return i + 2
        return i + 2  # ESC =, ESC >, etc.: skip

    def _csi(self, params: str, final: str):
        # private modes (\x1b[?1049h etc.): ignore
        if params.startswith("?"):
            return
        nums = [int(p) if p.isdigit() else 0 for p in params.split(";")] \
            if params else []

        def arg(k, default):
            return nums[k] if k < len(nums) and nums[k] else default

        if final == "m":
            self._sgr(params.split(";") if params else ["0"])
        elif final in "Hf":
            self.cur_r = min(max(arg(0, 1) - 1, 0), self.rows - 1)
            self.cur_c = min(max(arg(1, 1) - 1, 0), self.cols - 1)
        elif final == "A":
            self.cur_r = max(self.cur_r - arg(0, 1), 0)
        elif final == "B":
            self.cur_r = min(self.cur_r + arg(0, 1), self.rows - 1)
        elif final == "C":
            self.cur_c = min(self.cur_c + arg(0, 1), self.cols - 1)
        elif final == "D":
            self.cur_c = max(self.cur_c - arg(0, 1), 0)
        elif final == "G":
            self.cur_c = min(max(arg(0, 1) - 1, 0), self.cols - 1)
        elif final == "d":
            self.cur_r = min(max(arg(0, 1) - 1, 0), self.rows - 1)
        elif final == "J":
            self._erase_display(arg(0, 0) if nums else 0)
        elif final == "K":
            self._erase_line(arg(0, 0) if nums else 0)
        # else: scroll regions, modes, reports — brogue doesn't need them

    def _sgr(self, params: list[str]):
        i = 0
        while i < len(params):
            p = params[i]
            if p == "48" and i + 4 < len(params) and params[i + 1] == "2":
                self.bg = (int(params[i + 2] or 0), int(params[i + 3] or 0),
                           int(params[i + 4] or 0))
                i += 5
            elif p == "38" and i + 4 < len(params) and params[i + 1] == "2":
                i += 5  # foreground: not tracked
            elif p in ("0", "", "49"):
                self.bg = BLACK
                i += 1
            else:
                i += 1

    def _erase_display(self, mode: int):
        if mode == 0:
            self._erase_line(0)
            rng = range(self.cur_r + 1, self.rows)
        elif mode == 1:
            self._erase_line(1)
            rng = range(0, self.cur_r)
        else:
            rng = range(0, self.rows)
        for r in rng:
            self.chars[r] = [" "] * self.cols
            self.bgs[r] = [BLACK] * self.cols

    def _erase_line(self, mode: int):
        if mode == 0:
            cols = range(self.cur_c, self.cols)
        elif mode == 1:
            cols = range(0, self.cur_c + 1)
        else:
            cols = range(self.cols)
        for c in cols:
            self.chars[self.cur_r][c] = " "
            self.bgs[self.cur_r][c] = BLACK
