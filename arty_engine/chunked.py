LONGEST_SIZE_LINE = 1024
LONGEST_TRAILER_LINES = 64
HEX_DIGITS = set("0123456789abcdefABCDEF")
LINE_END = b"\r\n"


class ChunkedBodyError(ValueError):
    pass


class ChunkedBody:
    def __init__(self, stream):
        self.stream = stream
        self.left = 0
        self.finished = False

    def _line(self) -> bytes:
        line = self.stream.readline(LONGEST_SIZE_LINE + 1)
        if len(line) > LONGEST_SIZE_LINE:
            raise ChunkedBodyError("a chunk size line is too long")
        if not line.endswith(b"\n"):
            raise ChunkedBodyError("the chunked body ended early")
        return line

    def _next_size(self) -> int:
        size_text = self._line().split(b";", 1)[0].strip().decode("ascii", errors="replace")
        if not size_text or not set(size_text) <= HEX_DIGITS:
            raise ChunkedBodyError(f"{size_text!r} is not a chunk size")
        return int(size_text, 16)

    def _skip_trailers(self) -> None:
        for _ in range(LONGEST_TRAILER_LINES):
            if self._line().strip() == b"":
                return
        raise ChunkedBodyError("too many trailer lines after the chunked body")

    def read(self, size: int) -> bytes:
        if self.finished:
            return b""
        if self.left == 0:
            self.left = self._next_size()
            if self.left == 0:
                self._skip_trailers()
                self.finished = True
                return b""
        data = self.stream.read(min(size, self.left))
        if not data:
            raise ChunkedBodyError("the chunked body ended early")
        self.left -= len(data)
        if self.left == 0 and self.stream.read(len(LINE_END)) != LINE_END:
            raise ChunkedBodyError("a chunk did not end with CRLF")
        return data
