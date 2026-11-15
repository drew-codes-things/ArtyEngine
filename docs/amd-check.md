# Checking ArtyEngine on an AMD GPU

ArtyEngine's AMD support has **never run on a real AMD card**. `tools/amd-check.sh` is the check to run on a Linux
machine that has one (a rented cloud machine is fine). It starts from a fresh copy of this repository, uses a temporary
folder for everything it downloads, and removes it at the end. It writes `amd-check-report.txt` with a PASS or FAIL
line per step and the versions it found.

## What the machine needs

- An AMD GPU with the amdgpu driver loaded: `/dev/dri/renderD*` must exist. For the ROCm build `/dev/kfd` must
  exist too; without it the check uses the Vulkan build. AMD's list of cards ROCm supports is the compatibility
  matrix, https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html. Older cards such as the RX 570
  only work with the Vulkan build.
- Linux on x86-64, Python 3.10 or newer, `curl`, `tar`, `git`, `sha256sum` and `ldd`. `lspci`, `rocminfo`,
  `vulkaninfo` and `amd-smi` are used when they are there.
- For the Vulkan build: the Vulkan loader and Mesa's Vulkan driver (`libvulkan1` and `mesa-vulkan-drivers` on Debian
  and Ubuntu).
- About 4 GB of free disk for the runtime and the three models, plus AMD's ROCm packages when the machine has no ROCm
  10.0 of its own (several GB; their size was not measured).
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

Other options: `--backend vulkan` checks the Vulkan build on a machine that also has ROCm, `--stop-after
detect|download|config|start` stops early, `--engine-url http://127.0.0.1:<port>` runs only the model checks
against an engine you started yourself, `--no-rocm-wheels` refuses to install AMD's ROCm packages, and `--report
<file>` writes the report elsewhere. Set `HF_TOKEN` if Hugging Face limits anonymous downloads.

Exit codes: 0 every step passed, 1 a step failed, 2 no usable AMD GPU was found, 64 a wrong option.

## The steps

| Step | What it does | PASS means |
|---|---|---|
| `detect` | `lspci`, `/dev/kfd`, the amdgpu render node, `rocminfo` and `vulkaninfo` | an amdgpu render node is there; the backend is `rocm` with `/dev/kfd`, else `vulkan` |
| `download` | downloads release b12110 of the model runtime (ROCm 10.0 or Vulkan build) | its sha256 equals the digest the release lists (CHECK: the listing was unreachable, compare the printed sha256 with the README by hand) |
| `libraries` | `ldd` on the runtime's GPU backend library; for ROCm, installs AMD's `rocm[libraries]==10.0.0` packages into the temporary folder when the system lacks them | every library it needs loads |
| `devices` | asks the runtime which devices it sees | it lists a `ROCm0` or `Vulkan0` device, so models run on the card |
| `config` | writes an engine config on a free 127.0.0.1 port with `gpu_backend` and `runtimes` set | |
| `start` | starts the engine from this checkout | it answers, and its start line names the backend and the card's free memory |
| `pull-chat`, `pull-embed`, `pull-vision` | pulls `hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0`, `nomic-embed-text` and `hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M` through the engine | the pull succeeds |
| `chat`, `chat-cpu-only` | asks for the capital of France, on the card and with `num_gpu: 0` | the answer says Paris |
| `embed` | embeds a sentence | 768 numbers come back |
| `picture` | sends a generated picture of the text ROCK 42 | the answer reads ROCK and 42 |
| `loaded` | the engine's `/api/ps` | (information) |
| `memory`, `gpu-layers` | the free card memory the engine reads now, and the runtime's "offloaded N/M layers to GPU" lines | (information: all layers on the GPU is the goal) |
| `cleanup` | stops the engine and its runtimes, removes the temporary folder | |

## When a step fails

- `libraries` names the missing files. For ROCm, a system install of ROCm 10.0 or AMD's packages (the default) fix
  it; for Vulkan, the two packages above.
- `devices` lists nothing: check that your user is in the `render` and `video` groups and can open `/dev/kfd` and
  the render node. For a card outside AMD's list, the runtime's build notes suggest `HSA_OVERRIDE_GFX_VERSION` set to
  a supported version of the same family (for example `11.0.0` on RDNA3 or `10.3.0` on RDNA2); export it before
  running the check, since the engine and the runtime inherit it. Or try `--backend vulkan`.
- `start` fails: the report quotes the engine's last log lines.
- A wrong answer in `chat` or `picture` with every layer on the GPU points at the runtime build on that card, not at
  the engine; the same check with `--backend vulkan` tells the two apart.
