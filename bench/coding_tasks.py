from dataclasses import dataclass


@dataclass(frozen=True)
class CodingTask:
    name: str
    path: str
    source: str
    request: str
    check: str


INVENTORY = '''import json


def load_items(path):
    handle = open(path)
    data = json.load(handle)
    handle.close()
    return data


def total_value(items):
    total = 0
    for i in range(len(items) - 1):
        total += items[i]["price"] * items[i]["quantity"]
    return total


def cheapest(items):
    best = None
    for item in items:
        if best is None or item["price"] < best["price"]:
            best = item
    return best


def names(items):
    result = []
    for item in items:
        result.append(item["name"])
    return result


def restock(items, name, amount):
    for item in items:
        if item["name"] == name:
            item["quantity"] += amount
            return True
    return False
'''

TEMPERATURES = '''def celsius_to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32


def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9


def describe(celsius):
    if celsius < 0:
        return "freezing"
    if celsius < 15:
        return "cold"
    if celsius < 25:
        return "mild"
    return "hot"


def summary(readings):
    lines = []
    for reading in readings:
        lines.append(f"{reading}C is {describe(reading)} ({celsius_to_fahrenheit(reading):.1f}F)")
    return "\\n".join(lines)
'''

ROUTES = '''class Router:
    def __init__(self):
        self.routes = {}

    def add(self, method, path, handler):
        self.routes[(method, path)] = handler

    def dispatch(self, method, path, request):
        handler = self.routes.get((method, path))
        if handler is None:
            return 404, "not found"
        return 200, handler(request)


def hello(request):
    return "hello " + request.get("name", "world")


def echo(request):
    return request.get("text", "")


def build():
    router = Router()
    router.add("GET", "/hello", hello)
    router.add("POST", "/echo", echo)
    return router
'''

TASKS = [
    CodingTask("off_by_one", "inventory.py", INVENTORY,
               "total_value skips the last item. Fix it.",
               "from inventory import total_value\nassert total_value([{'price': 2, 'quantity': 3}, {'price': 5, 'quantity': 1}]) == 11\n"),
    CodingTask("with_open", "inventory.py", INVENTORY,
               "load_items should use a with block to open the file, and pass encoding='utf-8'.",
               "import inspect, inventory\nsource = inspect.getsource(inventory.load_items)\nassert 'with open(' in source and 'utf-8' in source\n"
               "import json, tempfile, os\npath = tempfile.mktemp()\nopen(path, 'w').write(json.dumps([1]))\nassert inventory.load_items(path) == [1]\nos.unlink(path)\n"),
    CodingTask("type_hints", "temperatures.py", TEMPERATURES,
               "Add type hints to every function in temperatures.py (floats for temperatures, list[float] for readings, str returns). "
               "Keep the behaviour the same.",
               "import inspect, temperatures as t\nfor f in (t.celsius_to_fahrenheit, t.fahrenheit_to_celsius, t.describe, t.summary):\n"
               "    assert inspect.signature(f).return_annotation is not inspect.Signature.empty, f.__name__\n"
               "assert t.summary([0, 30]) == '0C is cold (32.0F)\\n30C is hot (86.0F)'\nassert t.fahrenheit_to_celsius(212) == 100\n"),
    CodingTask("new_threshold", "temperatures.py", TEMPERATURES,
               "describe should return 'warm' between 20 and 25 degrees (20 included), and keep 'mild' for 15 up to 20.",
               "from temperatures import describe\nassert describe(20) == 'warm' and describe(24.9) == 'warm' and describe(19) == 'mild'\n"
               "assert describe(25) == 'hot' and describe(-1) == 'freezing' and describe(10) == 'cold'\n"),
    CodingTask("method_not_allowed", "routes.py", ROUTES,
               "dispatch should return 405 'method not allowed' when the path exists for another method, and still 404 for unknown paths.",
               "from routes import build\nr = build()\nassert r.dispatch('POST', '/hello', {}) == (405, 'method not allowed')\n"
               "assert r.dispatch('GET', '/nope', {}) == (404, 'not found')\nassert r.dispatch('GET', '/hello', {'name': 'x'}) == (200, 'hello x')\n"),
    CodingTask("rename_handler", "routes.py", ROUTES,
               "Rename the hello handler to greet everywhere in routes.py, and make it say 'hi ' instead of 'hello '.",
               "import routes\nassert not hasattr(routes, 'hello') and routes.greet({'name': 'a'}) == 'hi a'\n"
               "assert routes.build().dispatch('GET', '/hello', {}) == (200, 'hi world')\n"),
]
