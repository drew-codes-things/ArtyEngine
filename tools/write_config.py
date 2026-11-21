import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arty_engine.config import load_config

INSTALLED_FILE_MODE = 0o644


def site_config(source: Path, trainer_url: str, hosts: list[str], origins: list[str], roots: list[str],
                remote_models: Path | None = None) -> str:
    values = json.loads(source.read_text())
    if remote_models:
        values["remote_models"] = json.loads(remote_models.read_text())
    values["trainer_url"] = trainer_url
    if hosts:
        values["allowed_hosts"] = hosts
    if origins:
        values["allowed_origins"] = origins
    if roots:
        values["model_roots"] = roots
    return json.dumps(values, indent=2) + "\n"


def install(text: str, target: Path) -> None:
    partial = target.with_name(target.name + ".partial")
    partial.write_text(text)
    partial.chmod(INSTALLED_FILE_MODE)
    try:
        load_config(str(partial))
    except (ValueError, TypeError) as error:
        partial.unlink()
        raise ValueError(f"the engine would refuse this config: {error}") from error
    os.replace(partial, target)


def remote_names(target: Path) -> list[str]:
    return sorted(json.loads(target.read_text()).get("remote_models") or {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the engine config with this site's trainer controller")
    parser.add_argument("source", type=Path, help="the config to start from, such as deploy/engine.json")
    parser.add_argument("target", type=Path, help="where the engine reads its config")
    parser.add_argument("--trainer-url", required=True, help="the trainer controller's address, or an empty string for none")
    parser.add_argument("--allowed-host", action="append", default=[], help="a Host name to serve besides localhost (repeat for more)")
    parser.add_argument("--allowed-origin", action="append", default=[], help="a web page host to serve besides localhost (repeat for more)")
    parser.add_argument("--model-root", action="append", default=[], help="another engine model store to read in place (repeat for more)")
    parser.add_argument("--remote-models", type=Path, default=None,
                        help="a JSON file of outside OpenAI-compatible servers (remote_models), with no keys in it")
    arguments = parser.parse_args()
    try:
        install(site_config(arguments.source, arguments.trainer_url, arguments.allowed_host, arguments.allowed_origin,
                            arguments.model_root, arguments.remote_models), arguments.target)
    except (OSError, ValueError) as error:
        print(f"{arguments.target} not changed: {error}", file=sys.stderr)
        return 1
    print(f"{arguments.target}: trainer_url = {arguments.trainer_url or '(none)'}, "
          f"allowed_hosts = {arguments.allowed_host}, allowed_origins = {arguments.allowed_origin}, model_roots = {arguments.model_root}, "
          f"remote_models = {remote_names(arguments.target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
