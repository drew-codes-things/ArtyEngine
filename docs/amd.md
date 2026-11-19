# AMD GPUs (untested)

**No real AMD run has been done.** The AMD path is written and tested only against fakes (fake `amd-smi`, `rocm-smi`,
sysfs and device nodes) on machines without an AMD card. Treat it as a starting point; `tools/amd-check.sh`
([amd-check.md](amd-check.md)) checks it on a machine that has one.

## Runtime builds

The download links of the two builds and their sha256 digests are in the [README](../README.md#amd-gpus-untested). Link a `server` in each folder to its
model server binary (not the RPC server), as for CUDA.

- The **ROCm** build loads ROCm 10.0's `libamdhip64.so.7`, `libhipblas.so.3` and `librocblas.so.5`, which it does not
  include. Either install ROCm 10.0 on the system, or install AMD's own Python packages of it into a folder, the way
  the build itself was made. Name the card's kernel package as an extra, `device-<gfx target>` (such as
  `device-gfx1100` for an RX 7900 XTX; `rocminfo` or the `gfx_target_version` in
  `/sys/class/kfd/kfd/topology/nodes/*/properties` gives it), never the plain `device` extra, which falls back to
  gfx1010 on a machine without AMD's tools:
  `python3 -m venv /srv/ai/arty-engine/rocm-10.0 && /srv/ai/arty-engine/rocm-10.0/bin/pip install --no-cache-dir --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-gfx1100]==10.0.0"`.
  The libraries land in two folders, `_rocm_sdk_core/lib` (`libamdhip64.so.7`) and `_rocm_sdk_libraries/lib`
  (`libhipblas.so.3`, `librocblas.so.5`): list both in `runtimes.rocm.library_path`
  (`find /srv/ai/arty-engine/rocm-10.0 -name 'libamdhip64.so.7*' -o -name 'libhipblas.so.3*' -o -name 'librocblas.so.5*'`
  shows them). The gfx targets the build carries GPU code for are listed in the
  [README](../README.md#amd-gpus-untested).
- The **Vulkan** build needs the Vulkan loader (`libvulkan.so.1`) and a Vulkan driver for the card, on Debian
  `apt install libvulkan1 mesa-vulkan-drivers` (Mesa's RADV driver for AMD).

## Which cards ROCm supports

AMD's own list is the ROCm compatibility matrix,
https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html. On 15 Nov 2026 it covered ROCm 10.1.0 and
listed these Radeon cards: RX 9070 XT, RX 9070 GRE, RX 9070, RX 9060 XT LP, RX 9060 XT, RX 9060, RX 9050, RX 7900 XTX,
RX 7900 XT, RX 7900 GRE, RX 7800 XT, RX 7700 XT, RX 7700 and RX 7600, plus Radeon PRO (such as the W7900, W7800, W6800
and V620) and Instinct cards. Consumer RX 6000 cards are not on it. Check the page before choosing a card or a
rented machine; it changes between releases.

## Older cards need Vulkan

Cards such as the RX 570 (Polaris, gfx803) are neither on AMD's list nor among the GPU
code the ROCm build carries, so they run on the Vulkan build, or on the CPU.

## Config

Name both runtimes and let `auto` choose (it takes ROCm only for a card the ROCm build carries code for,
so an RX 570 lands on Vulkan), or set `gpu_backend` to `rocm` or `vulkan`. A card made to pass as another target with
`HSA_OVERRIDE_GFX_VERSION` needs `gpu_backend` set to `rocm`, since `auto` reads the card's own target:

```json
"gpu_backend": "auto",
"runtimes": {
  "rocm": {"runner_binary": "/srv/ai/arty-engine/runtime-rocm/server",
           "library_path": ["<venv>/lib/python3.X/site-packages/_rocm_sdk_core/lib",
                            "<venv>/lib/python3.X/site-packages/_rocm_sdk_libraries/lib"]},
  "vulkan": {"runner_binary": "/srv/ai/arty-engine/runtime-vulkan/server"}
}
```

## Container devices

ROCm needs `/dev/kfd` and the card's render node `/dev/dri/renderD*`; Vulkan needs only the
render node. Pass the card node `/dev/dri/card*` of the same GPU too. AMD's docs give GPU access to the `render` and
`video` groups, so give the nodes the container's own group ids: `render` for `/dev/kfd` and the render node, `video`
for the card node (`getent group render video` inside the container shows them). On Proxmox, for example:
`pct set <ct> -dev0 /dev/kfd,gid=<render> -dev1 /dev/dri/renderD128,gid=<render> -dev2 /dev/dri/card1,gid=<video>`.
The shipped unit runs the engine as root, which can open the nodes either way.

## Not done for AMD

`/api/ps` shows the engine's estimate as `size_vram` (there is no per-process reading as with
`nvidia-smi`). A `num_gpu: 0` request on the Vulkan build relies on `--device none` alone. When `amd-smi` lists several
cards, the engine reads the one with the most memory.
