# Benchmarking

`tools/container/` holds the bench helpers that run inside the container: `slice.sh` takes the GPU lock, asks the
controller for the card, unloads the engine's models, runs one bench and puts Arty's main model back; `slice.sh
--show-settings` prints what it would use. They expect to be copied into `/srv/ai/arty-engine`, beside `src`, with
`engine-test.json` from `deploy/` (the bench engine's config) and a `bench-support/` folder holding Arty's
`eval_scoring.py`, `e2e_battery.py` and `eval_images/` from the brain's `tests/` (the folder `bench.py --support`
reads). `bench-args.sh` also points at the brain's `/srv/ai/brain/system-prompt.md` and at the tool lists in this
repository's `tests/fixtures/`. `bench/bench.py` measures a server on load time, first token, generation speed,
Arty's chat battery, coding edits and vision.

`slice-engine.sh` starts the bench engine as `arty-engine-test` through `engine-restart.sh` and benches the address
in that config's `listen` (`127.0.0.1:11439` in `engine-test.json`, so the live engine on 11438 keeps serving). It
stops without benching when that port already answers before the start, when the unit is not running, or when the
unit never answers there; every bench config needs its own port and its own `state_dir`. The result file records
which models the live engine held before and after the bench (`live_engine`); `busy: true` means the two engines
shared the card and the numbers are not clean.

A Qwen3-0.6B draft model for `qwen3:14b` was slower than no speculation at a 24576 context on a 12 GB card, because
it pushes some of the 14B's layers off the GPU; n-gram speculation needs no extra memory and is what Arty runs.
