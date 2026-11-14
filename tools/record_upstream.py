import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arty_engine import translate
from arty_engine.config import load_config
from arty_engine.store import ModelStore

TIMEOUT_SECONDS = 900


def runner_port(runner_binary: str, model_name: str) -> int:
    listing = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, check=True).stdout
    for line in listing.splitlines():
        words = line.split()
        if words and words[0] == runner_binary and "-a" in words and words[words.index("-a") + 1] == model_name:
            return int(words[words.index("--port") + 1])
    raise SystemExit(f"no model runner is running {model_name}; load it through the engine first")


def record(port: int, body: dict) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        if not body.get("stream"):
            return {"json": json.loads(response.read())}
        return {"sse": [line.decode().strip() for line in response if line.strip()]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the model runner's own answers to the smoke requests, for the API tests")
    parser.add_argument("smoke_file", help="output of tools/smoke.py (its requests are replayed)")
    parser.add_argument("output")
    parser.add_argument("--config", default=None)
    parser.add_argument("--cases", default="judge_json,tool_round,tool_result,no_think_stream,think_unset")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    store = ModelStore(config.roots)
    smoke = json.loads(Path(arguments.smoke_file).read_text())
    recorded = {}
    for name in arguments.cases.split(","):
        request = smoke[name]["request"]
        model = store.get(request["model"])
        body = translate.chat_body(request, model, stream=request.get("stream", True) is not False)
        recorded[name] = {"body": body, **record(runner_port(config.runner_binary, model.name), body)}
        print(name, "recorded", file=sys.stderr)
    Path(arguments.output).write_text(json.dumps(recorded, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
