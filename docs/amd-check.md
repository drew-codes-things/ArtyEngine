# Checking ArtyEngine on an AMD GPU

ArtyEngine's AMD support has **never run on a real AMD card**. `tools/amd-check.sh` is the check to run on a Linux
machine that has one (a rented cloud machine is fine). It starts from a fresh copy of this repository, puts the
runtime, the models, AMD's packages and pip's and Mesa's caches in a temporary folder under `$TMPDIR` (`/var/tmp` when
that is not set, since `/tmp` is often in memory), and removes it at the end. With `--engine-url` the three models
are pulled into that engine's own store and stay there. It writes `amd-check-report.txt` with a PASS or FAIL line per
step, an overall result and the versions it found.

## What the machine needs

- An AMD GPU with the amdgpu driver loaded: `/dev/dri/renderD*` must exist. For the ROCm build `/dev/kfd` must
  exist too, and the card's gfx target (from `HSA_OVERRIDE_GFX_VERSION`, `rocminfo` or
  `/sys/class/kfd/kfd/topology/nodes/*/properties`) must be one the ROCm build carries code for; otherwise the check
  uses the Vulkan build. AMD's list of cards ROCm supports is the compatibility matrix,
  https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html. Older cards such as the RX 570 only
  work with the Vulkan build.
- Linux on x86-64, Python 3.10 or newer with its `venv` module (the `python3-venv` package on Debian and Ubuntu;
  the ROCm route checks it before downloading), `curl`, `tar`, `git`, `sha256sum` and `ldd`. `lspci`, `rocminfo`,
  `vulkaninfo` and `amd-smi` are used when they are there.
- For the Vulkan build: the Vulkan loader and Mesa's Vulkan driver (`libvulkan1` and `mesa-vulkan-drivers` on Debian
  and Ubuntu).
- Free disk where the temporary folder goes: about 5 GB for the ROCm build and about 3.5 GB for the Vulkan build.
  That is the runtime (ROCm: 0.26 GB download kept beside 1.25 GB unpacked), the three models (2.83 GB) and the
  512 MB the check's engine keeps free for pulls (`pull_keep_free_mb`); add AMD's ROCm packages when the machine
  has no ROCm 10.0 of its own (several GB; their size was not measured). Set `TMPDIR` to a folder on a disk with room.
- Internet access to GitHub (the runtime), Hugging Face (the models) and, for ROCm without a system install, AMD's
  package index.

## Driver setup (the only part that needs sudo)

Skip this when `ls -l /dev/kfd /dev/dri/` already lists the devices and your user can open them (many GPU cloud
images come ready). Otherwise:

1. Install the amdgpu driver and ROCm with AMD's guide for your distribution:
   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
2. Give your user access to the GPU, as AMD's guide says, then log out and in again:
   `sudo usermod -a -G video,render $LOGNAME`
3. For the Vulkan build: `sudo apt install libvulkan1 mesa-vulkan-drivers`
4. For AMD's ROCm packages: `sudo apt install python3-venv` (Debian and Ubuntu)

The check itself never uses sudo.

## Running it

```bash
git clone <your copy of the ArtyEngine repository> arty-engine
cd arty-engine
tools/amd-check.sh --dry-run
tools/amd-check.sh
cat amd-check-report.txt
```

`--dry-run` prints the plan and changes nothing. The full run takes as long as the downloads do. Send back
`amd-check-report.txt`; with `--keep` the temporary folder (runtime, models, engine and runtime logs) stays for a look,
and the report names it.

The report's `result:` line is `PASS` only when every step ran and passed, the runtime's logs showed layers on the
card for each model and none for the CPU chat. `FAIL` means a step failed. `INCOMPLETE` means the run was
interrupted, stopped early with `--stop-after`, or could not check the card use (with `--engine-url` the runtime's
logs are not reachable); its reason is on the same line.

Other options: `--backend vulkan` checks the Vulkan build on a machine that also has ROCm, `--stop-after
detect|download|config|start` stops early, `--engine-url http://127.0.0.1:<port>` runs only the model checks
against an engine you started yourself, `--no-rocm-wheels` refuses to install AMD's ROCm packages, and `--report
<file>` writes the report elsewhere. Set `HF_TOKEN` if Hugging Face limits anonymous downloads.

Exit codes: 0 no step failed (the `result:` line says whether the run was complete), 1 a step failed, 2 no usable
AMD GPU was found, 64 a wrong option, 130 interrupted.

## The steps

| Step | What it does | PASS means |
|---|---|---|
| `tools` | the programs above; for ROCm with AMD's packages, `python3 -m venv` | they are all there |
| `detect` | `lspci`, `/dev/kfd`, the amdgpu render node, the card's gfx target and `vulkaninfo` | an amdgpu render node is there; the backend is `rocm` with `/dev/kfd` and a target the ROCm build carries code for, else `vulkan` |
| `download` | downloads release b12110 of the model runtime (ROCm 10.0 or Vulkan build) | its sha256 equals the digest the release lists. When the listing can't be read the step fails, prints the sha256 to compare by hand with the README, and nothing is unpacked or run |
| `libraries` | `ldd` on the runtime's GPU backend library; for ROCm, installs AMD's `rocm[libraries,device-<gfx target>]==10.0.0` packages (the card's own kernel package, never a default one) into the temporary folder when the system lacks them, and uses both folders they put libraries in | every library it needs loads; the report names the device package installed |
| `devices` | asks the runtime which devices it sees | it lists a `ROCm0` device, or a `Vulkan0` device that is AMD's (an Intel or other Vulkan device alone fails), so models run on the card |
| `config` | writes an engine config on a free 127.0.0.1 port with `gpu_backend` and `runtimes` set | |
| `start` | starts the engine from this checkout | it answers, and its start line names the backend and the card's free memory with its source (`amd-smi`, `rocm-smi` or `sysfs`); memory it can't read fails the step |
| `pull-chat`, `pull-embed`, `pull-vision` | pulls `hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0`, `nomic-embed-text` and `hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M` through the engine | the pull succeeds |
| `chat`, `chat-cpu-only` | asks for the capital of France, on the card and with `num_gpu: 0` | the answer says Paris |
| `embed` | embeds a sentence | 768 numbers come back |
| `picture` | sends a generated picture of the text ROCK 42 | the answer holds the words ROCK and 42, whatever the punctuation |
| `offload-chat`, `offload-embed`, `offload-picture`, `offload-chat-cpu-only` | reads the "offloaded N/M layers to GPU" lines the runtime logged during that step | more than 0 layers on the card for each model; 0 for the CPU chat |
| `loaded` | the engine's `/api/ps` | (information) |
| `memory` | the free card memory the engine reads now | (information) |
| `cleanup` | stops the engine and its runtimes, removes the temporary folder | |

## When a step fails

- `libraries` names the missing files. For ROCm, a system install of ROCm 10.0 or AMD's packages (the default) fix
  it; AMD's packages need the card's gfx target, so an unreadable target fails here (set
  `HSA_OVERRIDE_GFX_VERSION`, see below). For Vulkan, the two packages above.
- `devices` lists nothing: check that your user is in the `render` and `video` groups and can open `/dev/kfd` and
  the render node. For a card outside AMD's list, the runtime's build notes suggest `HSA_OVERRIDE_GFX_VERSION` set to
  a supported version of the same family (for example `11.0.0` on RDNA3 or `10.3.0` on RDNA2); export it before
  running the check, since the engine and the runtime inherit it. Or try `--backend vulkan`.
- `start` fails: the report quotes the engine's last log lines.
- A wrong answer in `chat` or `picture` with every layer on the GPU points at the runtime build on that card, not at
  the engine; the same check with `--backend vulkan` tells the two apart.
