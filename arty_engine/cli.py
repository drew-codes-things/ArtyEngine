import sys

from . import runtime, server

COMMANDS = {"runtime": runtime.main}


def run(argv: list[str]) -> int:
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    server.main(argv)
    return 0


def entry() -> None:
    sys.exit(run(sys.argv[1:]))
