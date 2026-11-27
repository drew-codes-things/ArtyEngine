import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

TIMEOUT_SECONDS = 900
CALCULATE = {"type": "function", "function": {"name": "calculate", "description": "Work out arithmetic exactly.",
             "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}


def post(base: str, path: str, body: dict):
    request = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        if body.get("stream", True) is not False and path in ("/api/chat", "/api/generate", "/api/pull"):
            lines, first = [], None
            for line in response:
                if line.strip():
                    if first is None:
                        first = time.monotonic() - started
                    lines.append(json.loads(line))
            return {"stream": lines, "first_chunk_seconds": first, "seconds": time.monotonic() - started}
        return {"reply": json.loads(response.read()), "seconds": time.monotonic() - started}


def cases(model: str, vision_model: str, embed_model: str, image: str) -> list[tuple[str, str, dict]]:
    judge = {"model": model, "stream": False, "format": "json", "think": False, "keep_alive": -1,
             "options": {"temperature": 0, "num_predict": 120, "num_ctx": 12288},
             "messages": [{"role": "user", "content": 'Is "I have set a reminder" a claim that an action happened? Answer as JSON {"kind": "action" or "other"}.'}]}
    tool_round = {"model": model, "stream": False, "think": False, "keep_alive": -1, "tools": [CALCULATE],
                  "options": {"temperature": 0.4, "num_ctx": 12288, "num_predict": 2048, "repeat_penalty": 1.1},
                  "messages": [{"role": "system", "content": "You are Arty. Use tools for arithmetic."}, {"role": "user", "content": "what's 12 times 15?"}]}
    tool_result = {**tool_round, "messages": tool_round["messages"] + [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "calculate", "arguments": {"expression": "12*15"}}}]},
        {"role": "tool", "content": "180", "tool_name": "calculate"}]}
    thinking_stream = {"model": model, "stream": True, "think": True, "keep_alive": -1,
                       "options": {"temperature": 0.4, "num_ctx": 12288, "num_predict": 6144},
                       "messages": [{"role": "user", "content": "a train leaves at 14:50 and the trip takes 2 hours 25 minutes. when does it arrive? one line"}]}
    no_think_stream = {**thinking_stream, "think": False, "options": {"temperature": 0.4, "num_ctx": 12288, "num_predict": 2048}}
    think_unset = {"model": model, "stream": False, "keep_alive": -1, "options": {"num_ctx": 12288, "num_predict": 1024},
                   "messages": [{"role": "user", "content": "say hello in three words"}]}
    vision = {"model": vision_model, "stream": False, "keep_alive": "1m", "options": {"num_ctx": 12288, "num_predict": 2048},
              "messages": [{"role": "user", "content": "What is the invoice number and the total on this?", "images": [image]}]}
    embed = {"model": embed_model, "input": ["first sentence", "second sentence"], "keep_alive": "1m"}
    legacy_embed = {"model": embed_model, "prompt": "first sentence", "keep_alive": "1m"}
    unload_vision = {"model": vision_model, "keep_alive": 0}
    load_main = {"model": model, "keep_alive": -1}
    return [("judge_json", "/api/chat", judge), ("tool_round", "/api/chat", tool_round), ("tool_result", "/api/chat", tool_result),
            ("thinking_stream", "/api/chat", thinking_stream), ("no_think_stream", "/api/chat", no_think_stream),
            ("think_unset", "/api/chat", think_unset), ("vision", "/api/chat", vision), ("embed", "/api/embed", embed),
            ("legacy_embed", "/api/embeddings", legacy_embed), ("unload_vision", "/api/generate", unload_vision),
            ("load_main", "/api/generate", load_main)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Send brain-shaped requests to the engine's HTTP API and save what came back")
    parser.add_argument("base")
    parser.add_argument("output")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--vision-model", default="qwen2.5vl:7b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--image", required=True)
    parser.add_argument("--only", default="")
    arguments = parser.parse_args()
    image = base64.b64encode(Path(arguments.image).read_bytes()).decode()
    results = {}
    for name, path, body in cases(arguments.model, arguments.vision_model, arguments.embed_model, image):
        if arguments.only and name not in arguments.only.split(","):
            continue
        try:
            results[name] = {"path": path, "request": body, **post(arguments.base, path, body)}
        except Exception as error:
            results[name] = {"path": path, "request": body, "error": f"{type(error).__name__}: {error}"}
        outcome = results[name]
        print(name, f"{outcome.get('seconds', 0):.1f}s", outcome.get("error", ""), file=sys.stderr)
    for result in results.values():
        for message in result["request"].get("messages", []):
            if message.get("images"):
                message["images"] = ["<image>"]
    Path(arguments.output).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
