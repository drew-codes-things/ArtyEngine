import argparse
import json
import os
import re
import sys
from pathlib import Path

BACKEND_URLS = {"engine": "http://127.0.0.1:11438"}
ENGINE_URL_SETTING = "arty_engine_url"
SYSTEMD_FOLDER = Path("/etc/systemd/system")
DROP_IN_NAME = "settings.conf"
UNIT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9@._-]*")
MODULE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


class SwitchError(Exception):
    pass


def drop_in_path(unit: str) -> Path:
    return SYSTEMD_FOLDER / f"{unit}.service.d" / DROP_IN_NAME


def restart_hint(unit: str) -> str:
    return f"systemctl daemon-reload && systemctl restart {unit}"


def check_names(unit: str, module: str) -> None:
    if not UNIT_PATTERN.fullmatch(unit) or ".." in unit:
        raise SwitchError(f"{unit!r} is not a systemd unit name")
    if not MODULE_PATTERN.fullmatch(module):
        raise SwitchError(f"{module!r} is not a Python module name")


def drop_in_text(python: str, module: str, settings: Path) -> str:
    return f"[Service]\nExecStart=\nExecStart={python} -m {module} --settings {settings}\n"


def read_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        values = json.loads(path.read_text())
    except ValueError as error:
        raise SwitchError(f"{path} is not valid JSON ({error}); fix it by hand first") from error
    if not isinstance(values, dict):
        raise SwitchError(f"{path} should hold a JSON object of controller settings")
    return values


def write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def check_drop_in(drop_in: Path, settings: Path) -> bool:
    if not drop_in.exists():
        return False
    if f"--settings {settings}\n" not in drop_in.read_text():
        raise SwitchError(f"{drop_in} starts the controller with other settings than {settings}:\n{drop_in.read_text()}")
    return True


def switch(backend: str, settings: Path, drop_in: Path, python: str, module: str) -> list[str]:
    values = read_settings(settings)
    drop_in_present = check_drop_in(drop_in, settings)
    values[ENGINE_URL_SETTING] = BACKEND_URLS[backend]
    write_atomically(settings, json.dumps(values, indent=2) + "\n")
    done = [f"{settings}: {ENGINE_URL_SETTING} = {values[ENGINE_URL_SETTING]}"]
    if not drop_in_present:
        write_atomically(drop_in, drop_in_text(python, module, settings))
        done.append(f"{drop_in}: controller now reads {settings}")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Point the trainer controller at the engine")
    parser.add_argument("backend", choices=sorted(BACKEND_URLS))
    parser.add_argument("--settings", required=True, help="the controller's JSON settings file")
    parser.add_argument("--unit", required=True, help="the controller's systemd unit, without .service")
    parser.add_argument("--module", required=True, help="the Python module that runs the controller")
    parser.add_argument("--drop-in", default=None, help="the drop-in to write (default: settings.conf in the unit's .d folder)")
    parser.add_argument("--python", required=True, help="the Python that runs the controller")
    arguments = parser.parse_args()
    try:
        check_names(arguments.unit, arguments.module)
        drop_in = Path(arguments.drop_in) if arguments.drop_in else drop_in_path(arguments.unit)
        done = switch(arguments.backend, Path(arguments.settings), drop_in, arguments.python, arguments.module)
    except (SwitchError, OSError) as error:
        print(f"controller settings not changed: {error}", file=sys.stderr)
        return 1
    print("\n".join(done))
    print(f"now run: {restart_hint(arguments.unit)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
