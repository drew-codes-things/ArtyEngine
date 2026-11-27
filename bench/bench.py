import argparse
import ast
import base64
import json
import operator
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arty_engine import gpu
from coding_tasks import TASKS

TIMEOUT_SECONDS = 900
DEFAULT_CONTEXT_TOKENS = 24576
BRAIN_OPTIONS = {"temperature": 0.4, "num_predict": 2048, "repeat_penalty": 1.1}
CODE_OPTIONS = {"temperature": 0, "seed": 7, "num_predict": 2048}
MAX_TOOL_ROUNDS = 4
CHECK_TIMEOUT_SECONDS = 20
CHECK_ROOT = "/run/arty-bench-checks"
SANDBOX_START_SECONDS = 10
SANDBOX_FOLDER = "/check"
SANDBOX_PROPERTIES = ("DynamicUser=yes", "PrivateNetwork=yes", "ProtectSystem=strict", "ProtectHome=yes",
                      "InaccessiblePaths=/srv/ai", "NoNewPrivileges=yes", "MemoryMax=512M",
                      f"RuntimeMaxSec={CHECK_TIMEOUT_SECONDS}")
SANDBOX_PROBE = """import os
import socket
assert os.getuid() != 0, "the check runs as root"
assert not os.access("/srv/ai", os.R_OK), "/srv/ai is readable"
try:
    socket.create_connection(("127.0.0.1", 11434), timeout=2).close()
except OSError:
    pass
else:
    raise SystemExit("the network is reachable")
"""
VRAM_SETTLE_SECONDS = 30
SPEED_REPEATS = 2
STORY_TOKENS = 400
CODE_SYSTEM = ("You are a coding assistant working in a project. Change files only with edit_file: copy old_text exactly from the "
               "file as shown (without line numbers), with 2 or 3 lines around the change, and put the whole replacement in new_text. "
               "Make every change the request needs, then reply with one short sentence.")
FAKE_TOOL_RESULTS = {
    "weather": "Tomorrow: light rain from 14:00, 70% chance of rain, 12C.",
    "calendar_events": "Nothing on the calendar this week.",
    "deadlines_show": "No deadlines.",
    "schedule_task": "Scheduled reminder #7 in 3 hours: {text}",
    "calendar_add": "Added to the calendar: {text}",
    "list_add": "Added to the list: {text}",
    "draft": "Draft saved.",
}
ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
              ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def call(base: str, path: str, body: dict | None = None, timeout: int = TIMEOUT_SECONDS):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{base}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def stream(base: str, body: dict) -> dict:
    request = urllib.request.Request(f"{base}/api/chat", data=json.dumps({**body, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    first = None
    content, last = [], {}
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        for line in response:
            if not line.strip():
                continue
            last = json.loads(line)
            if last.get("error"):
                raise RuntimeError(last["error"])
            piece = last.get("message") or {}
            if first is None and (piece.get("content") or piece.get("thinking")):
                first = time.monotonic() - started
            content.append(piece.get("content") or "")
    return {**last, "first_token_seconds": first, "wall_seconds": time.monotonic() - started, "content": "".join(content)}


def vram_used_mb() -> int | None:
    memory = gpu.any_card_memory()
    return memory.used_mb if memory else None


def vram_growth_mb(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def unload_all(base: str) -> None:
    for model in call(base, "/api/ps").get("models", []):
        call(base, "/api/generate", {"model": model["name"], "keep_alive": 0})
    deadline = time.monotonic() + VRAM_SETTLE_SECONDS
    while call(base, "/api/ps").get("models") and time.monotonic() < deadline:
        time.sleep(0.5)
    time.sleep(2)


def tokens_per_second(reply: dict) -> float:
    duration = reply.get("eval_duration") or 0
    return round(reply.get("eval_count", 0) / (duration / 1e9), 1) if duration else 0.0


def prompt_tokens_per_second(reply: dict) -> float:
    duration = reply.get("prompt_eval_duration") or 0
    return round(reply.get("prompt_eval_count", 0) / (duration / 1e9), 1) if duration else 0.0


def long_prompt(system_prompt: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": "what's on my calendar this week?"}]


def never() -> bool:
    return False


def speed(base: str, model: str, system_prompt: str, tools: list, context: int = DEFAULT_CONTEXT_TOKENS) -> dict:
    brain_options = {**BRAIN_OPTIONS, "num_ctx": context}
    unload_all(base)
    baseline = vram_used_mb()
    started = time.monotonic()
    call(base, "/api/generate", {"model": model, "keep_alive": -1, "options": {"num_ctx": context}})
    load_seconds = time.monotonic() - started
    vram = vram_growth_mb(baseline, vram_used_mb())
    short = [stream(base, {"model": model, "think": False, "keep_alive": -1, "options": {**brain_options, "num_predict": 16},
                           "messages": [{"role": "user", "content": f"say hi ({attempt})"}]}) for attempt in range(SPEED_REPEATS)]
    long_cold = stream(base, {"model": model, "think": False, "keep_alive": -1, "tools": tools, "options": {**brain_options, "num_predict": 64},
                              "messages": long_prompt(system_prompt)})
    long_warm = stream(base, {"model": model, "think": False, "keep_alive": -1, "tools": tools, "options": {**brain_options, "num_predict": 64},
                              "messages": long_prompt(system_prompt)})
    stories = [call(base, "/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": -1,
                                        "options": {"temperature": 0, "seed": 1, "num_ctx": context, "num_predict": STORY_TOKENS},
                                        "messages": [{"role": "user", "content": "Write a 500-word story about a lighthouse keeper."}]})
               for _ in range(SPEED_REPEATS)]
    return {
        "load_seconds": round(load_seconds, 2), "vram_mb": vram,
        "first_token_short_seconds": round(statistics.median(item["first_token_seconds"] or 0 for item in short), 3),
        "first_token_long_cold_seconds": round(long_cold["first_token_seconds"] or long_cold["wall_seconds"], 3),
        "first_token_long_warm_seconds": round(long_warm["first_token_seconds"] or long_warm["wall_seconds"], 3),
        "long_prompt_tokens": long_cold.get("prompt_eval_count", 0), "prompt_tokens_per_second": prompt_tokens_per_second(long_cold),
        "generate_tokens_per_second": round(statistics.median(tokens_per_second(story) for story in stories), 1),
        "story_tokens": stories[-1].get("eval_count", 0),
    }


def safe_arithmetic(expression: str) -> str:
    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
            return ARITHMETIC[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ARITHMETIC:
            return ARITHMETIC[type(node.op)](evaluate(node.operand))
        raise ValueError("not arithmetic")
    text = str(expression).replace("^", "**").replace("x", "*").replace("×", "*")
    if "% of" in text:
        percent, whole = text.split("% of", 1)
        return str(float(percent) / 100 * float(whole))
    try:
        return str(evaluate(ast.parse(text, mode="eval")))
    except (ValueError, SyntaxError, ZeroDivisionError):
        return "calculate couldn't read that expression"


def fake_tool(name: str, arguments) -> str:
    arguments = arguments if isinstance(arguments, dict) else {}
    if name == "calculate":
        return safe_arithmetic(arguments.get("expression", ""))
    template = FAKE_TOOL_RESULTS.get(name, "Done.")
    return template.format(text=json.dumps(arguments)[:200])


def chat_case(base: str, model: str, system_prompt: str, tools: list, message: str, context: int = DEFAULT_CONTEXT_TOKENS):
    from eval_scoring import Outcome
    history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    used, structured, started = [], 0, time.monotonic()
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            reply = call(base, "/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": -1, "tools": tools,
                                             "options": {**BRAIN_OPTIONS, "num_ctx": context}, "messages": history})
            message_out = reply["message"]
            calls = message_out.get("tool_calls") or []
            if not calls:
                return Outcome(message_out.get("content") or "", tuple(used), time.monotonic() - started), structured
            history.append({"role": "assistant", "content": message_out.get("content") or "", "tool_calls": calls})
            for tool_call in calls:
                function = tool_call.get("function") or {}
                used.append(function.get("name", ""))
                structured += isinstance(function.get("arguments"), dict)
                history.append({"role": "tool", "content": fake_tool(function.get("name", ""), function.get("arguments")),
                                "tool_name": function.get("name", "")})
        return Outcome("", tuple(used), time.monotonic() - started, "ran out of tool rounds"), structured
    except Exception as error:
        return Outcome("", tuple(used), time.monotonic() - started, f"{type(error).__name__}: {error}"), structured


def chat_eval(base: str, model: str, system_prompt: str, tools: list, context: int = DEFAULT_CONTEXT_TOKENS,
              stop: Callable[[], bool] = never) -> dict:
    from e2e_battery import CHAT_CASES
    from eval_scoring import score, summarise
    native = {tool["function"]["name"] for tool in tools}
    results, skipped, structured_calls, stopped_early = [], [], 0, False
    for case in CHAT_CASES:
        if case.tools and not set(case.tools) & native:
            skipped.append(case.message)
            continue
        if stop():
            stopped_early = True
            break
        outcome, structured = chat_case(base, model, system_prompt, tools, case.message, context)
        structured_calls += structured
        results.append(score(case, outcome))
    calls = sum(len(result["tools_used"]) for result in results)
    return {"summary": summarise(results), "results": results, "skipped_needs_hub_tools": skipped,
            "tool_calls": calls, "tool_calls_with_object_arguments": structured_calls, "stopped_early": stopped_early}


def numbered(source: str) -> str:
    return "\n".join(f"{number:>4}\t{line}" for number, line in enumerate(source.splitlines(), 1))


def apply_edits(source: str, calls: list) -> tuple[str, str]:
    for tool_call in calls:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") if isinstance(function.get("arguments"), dict) else {}
        if function.get("name") != "edit_file":
            return source, f"called {function.get('name')}"
        old, new = arguments.get("old_text", ""), arguments.get("new_text", "")
        count = source.count(old) if old else 0
        if count == 0:
            return source, "old_text not found"
        if count > 1 and not arguments.get("replace_all"):
            return source, "old_text matched more than once"
        source = source.replace(old, new) if arguments.get("replace_all") else source.replace(old, new, 1)
    return source, ""


def check_command(folder: str) -> list[str]:
    properties = [*SANDBOX_PROPERTIES, f"BindReadOnlyPaths={folder}:{SANDBOX_FOLDER}"]
    return ["systemd-run", "--wait", "--pipe", "--quiet", "--collect", f"--working-directory={SANDBOX_FOLDER}",
            "-E", "PYTHONDONTWRITEBYTECODE=1", *[part for value in properties for part in ("-p", value)],
            sys.executable, "-E", "-s", "check.py"]


def run_check(path: str, source: str, check: str) -> str:
    Path(CHECK_ROOT).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CHECK_ROOT) as folder:
        Path(folder).chmod(0o755)
        Path(folder, path).write_text(source)
        Path(folder, "check.py").write_text(check)
        for written in (Path(folder, path), Path(folder, "check.py")):
            written.chmod(0o644)
        try:
            finished = subprocess.run(check_command(folder), capture_output=True, text=True,
                                      timeout=CHECK_TIMEOUT_SECONDS + SANDBOX_START_SECONDS)
        except subprocess.TimeoutExpired:
            return "check timed out"
        return "" if finished.returncode == 0 else (finished.stderr.strip().splitlines() or ["check failed"])[-1]


def sandbox_problem() -> str:
    return run_check("probe.py", "", SANDBOX_PROBE)


def code_eval(base: str, model: str, edit_tools: list, context: int = DEFAULT_CONTEXT_TOKENS, stop: Callable[[], bool] = never) -> dict:
    results, stopped_early = [], False
    for task in TASKS:
        if stop():
            stopped_early = True
            break
        source, history = task.source, [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content":
                   f"{task.request}\n\n{task.path}:\n{numbered(task.source)}"}]
        started, generated, generating_seconds, problem = time.monotonic(), 0, 0.0, ""
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                reply = call(base, "/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": -1, "tools": edit_tools,
                                                 "options": {**CODE_OPTIONS, "num_ctx": context}, "messages": history})
                generated += reply.get("eval_count", 0)
                generating_seconds += (reply.get("eval_duration") or 0) / 1e9
                calls = reply["message"].get("tool_calls") or []
                if not calls:
                    break
                source, problem = apply_edits(source, calls)
                history.append({"role": "assistant", "content": reply["message"].get("content") or "", "tool_calls": calls})
                for tool_call in calls:
                    history.append({"role": "tool", "content": problem or "Edited.", "tool_name": "edit_file"})
                if problem:
                    break
        except Exception as error:
            problem = f"{type(error).__name__}: {error}"
        failure = problem or ("no edit made" if source == task.source else run_check(task.path, source, task.check))
        results.append({"task": task.name, "passed": not failure, "failure": failure, "seconds": round(time.monotonic() - started, 1),
                        "generated_tokens": generated, "tokens_per_second": round(generated / generating_seconds, 1) if generating_seconds else 0.0})
    passed = sum(result["passed"] for result in results)
    tokens = sum(result["generated_tokens"] for result in results)
    seconds = sum(result["seconds"] for result in results)
    speeds = [result["tokens_per_second"] for result in results]
    return {"passed": passed, "cases": len(results), "total_seconds": round(seconds, 1), "generated_tokens": tokens,
            "median_tokens_per_second": statistics.median(speeds) if speeds else 0.0, "results": results,
            "stopped_early": stopped_early}


def vision_eval(base: str, model: str, images: Path, context: int = DEFAULT_CONTEXT_TOKENS, stop: Callable[[], bool] = never) -> dict:
    from e2e_battery import VISION_CASES
    from eval_scoring import Outcome, score, summarise
    results, stopped_early = [], False
    for case in VISION_CASES:
        if stop():
            stopped_early = True
            break
        picture = base64.b64encode((images / case.image).read_bytes()).decode()
        started = time.monotonic()
        try:
            reply = call(base, "/api/chat", {"model": model, "stream": False, "keep_alive": "1m", "options": {"num_ctx": context, "num_predict": 2048},
                                             "messages": [{"role": "user", "content": case.message, "images": [picture]}]})
            outcome = Outcome(reply["message"].get("content") or "", (), time.monotonic() - started)
        except Exception as error:
            outcome = Outcome("", (), time.monotonic() - started, f"{type(error).__name__}: {error}")
        results.append(score(case, outcome))
    return {"summary": summarise(results), "results": results, "stopped_early": stopped_early}


def stop_file_appeared(stop_file: str) -> bool:
    return bool(stop_file) and Path(stop_file).exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark of a model server for Arty that speaks the engine's HTTP API")
    parser.add_argument("base", help="for example http://127.0.0.1:11438")
    parser.add_argument("label")
    parser.add_argument("--parts", default="speed,chat,code,vision")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--code-models", default="qwen3:14b,hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS")
    parser.add_argument("--vision-model", default="qwen2.5vl:7b")
    parser.add_argument("--speed-models", default="qwen3:14b,hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS")
    parser.add_argument("--support", required=True, help="folder with eval_scoring.py, e2e_battery.py and eval_images/")
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--tools", required=True, help="the brain's native tool definitions (JSON list)")
    parser.add_argument("--edit-tools", required=True, help="the CLI's read_file and edit_file definitions (JSON list)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_CONTEXT_TOKENS, help="context size sent with every request")
    parser.add_argument("--stop-file", default="", help="when this file appears, finish the case in progress and stop")
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.support)
    system_prompt = Path(arguments.system_prompt).read_text()
    tools = json.loads(Path(arguments.tools).read_text())
    edit_tools = json.loads(Path(arguments.edit_tools).read_text())
    output = Path(arguments.output)
    report = json.loads(output.read_text()) if output.exists() else {"label": arguments.label, "base": arguments.base}
    parts = arguments.parts.split(",")

    def save(key, value):
        report[key] = value
        output.write_text(json.dumps(report, indent=1))
        print(key, json.dumps(value)[:400], file=sys.stderr, flush=True)

    base, context = arguments.base, arguments.num_ctx
    stop = partial(stop_file_appeared, arguments.stop_file)
    save("num_ctx", context)
    jobs = []
    if "speed" in parts:
        jobs += [(f"speed:{model}", partial(speed, base, model, system_prompt, tools, context)) for model in arguments.speed_models.split(",")]
    if "chat" in parts:
        jobs.append((f"chat:{arguments.model}", partial(chat_eval, base, arguments.model, system_prompt, tools, context, stop)))
    if "code" in parts:
        problem = sandbox_problem()
        if problem:
            save("code", {"skipped": f"sandbox check failed: {problem}"})
        else:
            jobs += [(f"code:{model}", partial(code_eval, base, model, edit_tools, context, stop)) for model in arguments.code_models.split(",")]
    if "vision" in parts:
        images = Path(arguments.support) / "eval_images"
        jobs.append((f"vision:{arguments.vision_model}", partial(vision_eval, base, arguments.vision_model, images, context, stop)))
    for key, run in jobs:
        if stop():
            break
        save(key, run())
    if stop():
        save("stopped", f"stop file {arguments.stop_file} appeared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
