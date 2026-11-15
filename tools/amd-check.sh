#!/bin/bash
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
STEPS=$REPO/tools/amd_check_steps.py
RELEASE=b12110
RELEASE_API=https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$RELEASE
ASSET_URL_PREFIX=https://github.com/ggml-org/llama.cpp/releases/download/$RELEASE/llama-$RELEASE-bin-
ROCM_ASSET_KIND=ubuntu-rocm-10.0-x64
VULKAN_ASSET_KIND=ubuntu-vulkan-x64
ROCM_WHEEL_INDEX=https://stable.repo.amd.com/rocm/whl-next/
ROCM_WHEEL_VERSION=10.0.0
CHAT_MODEL=hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0
EMBED_MODEL=nomic-embed-text
VISION_MODEL=hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M
DRM=${AMD_CHECK_DRM:-/sys/class/drm}
DEV=${AMD_CHECK_DEV:-/dev}
ENGINE_START_SECONDS=60
STOP_SECONDS=30
EXIT_FAILED=1
EXIT_NO_AMD=2
EXIT_USAGE=64
STEP_ORDER="detect download config start"

DRY_RUN=false
KEEP=false
INSTALL_ROCM_WHEELS=true
BACKEND=""
STOP_AFTER=""
ENGINE_URL=""
REPORT=$PWD/amd-check-report.txt
WORK=""
ENGINE_PID=""
ASSET=""
RUNNER=""
LIBRARIES=""
GPU_SUMMARY=""
ROCM_LIBRARIES="system"
FAILED=false
RESULTS=()

usage() {
  cat >&2 <<EOF
usage: $0 [--dry-run] [--backend rocm|vulkan] [--stop-after detect|download|config|start]
          [--engine-url URL] [--report FILE] [--keep] [--no-rocm-wheels]
  --dry-run          print the plan and change nothing
  --backend          use this backend instead of the one the card suggests
  --stop-after       stop after that step (then clean up and write the report)
  --engine-url       check an engine that is already running instead of starting one
  --report           where the report goes (default: ./amd-check-report.txt)
  --keep             keep the temporary folder (runtime, models, logs) for a look afterwards
  --no-rocm-wheels   do not install AMD's ROCm $ROCM_WHEEL_VERSION libraries when the system lacks them
EOF
  exit $EXIT_USAGE
}

parse_arguments() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=true ;;
      --keep) KEEP=true ;;
      --no-rocm-wheels) INSTALL_ROCM_WHEELS=false ;;
      --backend) [ $# -ge 2 ] || usage; BACKEND=$2; shift ;;
      --stop-after) [ $# -ge 2 ] || usage; STOP_AFTER=$2; shift ;;
      --engine-url) [ $# -ge 2 ] || usage; ENGINE_URL=${2%/}; shift ;;
      --report) [ $# -ge 2 ] || usage; REPORT=$2; shift ;;
      -h | --help) usage ;;
      *) echo "unknown option: $1" >&2; usage ;;
    esac
    shift
  done
  case "$BACKEND" in "" | rocm | vulkan) ;; *) echo "--backend must be rocm or vulkan" >&2; usage ;; esac
  case " $STEP_ORDER " in *" $STOP_AFTER "*) ;; *) [ -z "$STOP_AFTER" ] || { echo "--stop-after must be one of: $STEP_ORDER" >&2; usage; } ;; esac
}

asset_url() {
  echo "$ASSET_URL_PREFIX$1.tar.gz"
}

print_plan() {
  cat <<EOF
AMD check plan (dry run: nothing is downloaded, started or written)
  1. detect the GPU: lspci, $DEV/kfd, amdgpu render nodes in $DRM, rocminfo and vulkaninfo when installed
  2. download release $RELEASE of the model runtime into a temporary folder and check its sha256:
       ROCm build when $DEV/kfd is there: $(asset_url $ROCM_ASSET_KIND)
       Vulkan build otherwise: $(asset_url $VULKAN_ASSET_KIND)
     ROCm only: when the system has no ROCm $ROCM_WHEEL_VERSION libraries, install AMD's rocm[libraries]==$ROCM_WHEEL_VERSION
     packages from $ROCM_WHEEL_INDEX into that folder, as your own user; then ask the runtime which devices it sees
  3. write a temporary engine config listening on a free 127.0.0.1 port
  4. start the engine from $REPO
  5. pull through the engine, from Hugging Face: $CHAT_MODEL, $EMBED_MODEL, $VISION_MODEL
  6. chat check (expects Paris), the same chat on the CPU (num_gpu 0), embed check (768 numbers),
     picture check on a generated picture of the text ROCK 42
  7. record the free card memory the engine reads, and the runtime's GPU layer count
  8. write $REPORT
  9. stop the engine and remove the temporary folder
EOF
}

record() {
  local status=$1 step=$2 detail=$3
  RESULTS+=("$status $step: $detail")
  echo "$status $step: $detail"
  [ "$status" != FAIL ] || FAILED=true
}

stops_after() {
  [ "$STOP_AFTER" = "$1" ]
}

amd_render_node() {
  local node
  for node in "$DRM"/renderD*; do
    [ -e "$node/device/driver" ] || continue
    if [ "$(basename "$(readlink -f "$node/device/driver")")" = amdgpu ]; then
      basename "$node"
      return 0
    fi
  done
  return 1
}

amd_pci_lines() {
  lspci -nn 2>/dev/null | grep -Ei 'vga|3d|display' | grep -Ei '\[1002:|advanced micro devices|ati ' || true
}

detect_gpu() {
  local pci render kfd=no gfx vulkan_device
  pci=$(amd_pci_lines)
  render=$(amd_render_node) || render=""
  [ -e "$DEV/kfd" ] && kfd=yes
  gfx=$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)
  vulkan_device=$(vulkaninfo --summary 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= *//' || true)
  GPU_SUMMARY="lspci: ${pci:-no AMD display device}; render node: ${render:-none}; /dev/kfd: $kfd; rocminfo: ${gfx:-not installed or no gfx}; vulkaninfo: ${vulkan_device:-not installed or no device}"
  if [ -z "$render" ]; then
    if [ -n "$pci" ]; then
      record FAIL detect "an AMD card is in lspci but the amdgpu driver gives no render node in $DRM (install the driver, see docs/amd-check.md). $GPU_SUMMARY"
    else
      record FAIL detect "no AMD GPU found. $GPU_SUMMARY"
    fi
    return $EXIT_NO_AMD
  fi
  if [ ! -e "$DEV/dri/$render" ]; then
    record FAIL detect "$render is in $DRM but not in $DEV/dri (a container needs it passed in). $GPU_SUMMARY"
    return $EXIT_NO_AMD
  fi
  if [ -z "$BACKEND" ]; then
    if [ $kfd = yes ]; then BACKEND=rocm; else BACKEND=vulkan; fi
  fi
  record PASS detect "backend $BACKEND. $GPU_SUMMARY"
}

published_digest() {
  curl -fsSL --retry 3 -o "$WORK/release.json" "$RELEASE_API" 2>/dev/null || return 1
  python3 "$STEPS" published-digest "$WORK/release.json" "$ASSET"
}

download_runtime() {
  local kind=$ROCM_ASSET_KIND url published actual
  [ "$BACKEND" = vulkan ] && kind=$VULKAN_ASSET_KIND
  url=$(asset_url "$kind")
  ASSET=${url##*/}
  curl -fL --retry 3 -sS -o "$WORK/runtime.tar.gz" "$url" || { record FAIL download "could not download $url"; return 1; }
  actual=$(sha256sum "$WORK/runtime.tar.gz" | cut -d' ' -f1)
  if ! published=$(published_digest); then
    published=""
    record CHECK download "$ASSET has sha256 $actual; the release listing gave no digest, so compare it by hand with README.md (AMD GPUs (untested)) or the release page"
  elif [ "$actual" != "$published" ]; then
    record FAIL download "$ASSET has sha256 $actual, but the release lists $published"
    return 1
  fi
  mkdir -p "$WORK/runtime"
  tar -xzf "$WORK/runtime.tar.gz" -C "$WORK/runtime" --strip-components=1 || { record FAIL download "could not unpack $ASSET"; return 1; }
  RUNNER=$(find "$WORK/runtime" -maxdepth 1 -type f -name '*-server' ! -name '*rpc*' | head -1)
  [ -n "$RUNNER" ] || { record FAIL download "no model server binary in $ASSET"; return 1; }
  [ -z "$published" ] || record PASS download "$ASSET, sha256 $actual matches the published digest"
}

missing_libraries() {
  LD_LIBRARY_PATH="$1" ldd "$2" 2>&1 | grep 'not found' | awk '{print $1}' | paste -sd' ' -
}

install_rocm_wheels() {
  local found
  python3 -m venv "$WORK/rocm-libs" || return 1
  "$WORK/rocm-libs/bin/pip" install -q --index-url "$ROCM_WHEEL_INDEX" "rocm[libraries]==$ROCM_WHEEL_VERSION" || return 1
  found=$(find "$WORK/rocm-libs" -name 'libamdhip64.so.7*' | head -1)
  [ -n "$found" ] || return 1
  dirname "$found"
}

check_libraries() {
  local backend_library missing wheel_folder
  LIBRARIES=""
  backend_library=$(find "$WORK/runtime" -maxdepth 1 -name "lib*-${BACKEND/rocm/hip}.so" | head -1)
  [ -n "$backend_library" ] || { record FAIL libraries "no $BACKEND backend library in $ASSET"; return 1; }
  missing=$(missing_libraries "$WORK/runtime" "$backend_library")
  if [ -n "$missing" ] && [ "$BACKEND" = rocm ] && $INSTALL_ROCM_WHEELS; then
    echo "installing AMD's ROCm $ROCM_WHEEL_VERSION libraries into $WORK/rocm-libs (a large download)"
    wheel_folder=$(install_rocm_wheels) || { record FAIL libraries "missing $missing, and AMD's ROCm packages did not install"; return 1; }
    LIBRARIES=$wheel_folder
    ROCM_LIBRARIES="AMD packages rocm[libraries]==$ROCM_WHEEL_VERSION"
    missing=$(missing_libraries "$LIBRARIES:$WORK/runtime" "$backend_library")
  fi
  if [ -n "$missing" ]; then
    record FAIL libraries "the $BACKEND build cannot load: $missing (see docs/amd-check.md)"
    return 1
  fi
  record PASS libraries "every library of $(basename "$backend_library") loads (ROCm libraries: $ROCM_LIBRARIES)"
}

check_devices() {
  local listed prefix=ROCm
  [ "$BACKEND" = vulkan ] && prefix=Vulkan
  listed=$(LD_LIBRARY_PATH="${LIBRARIES:+$LIBRARIES:}$WORK/runtime" "$RUNNER" --list-devices 2>&1 | grep -E "^ *${prefix}[0-9]+:" | paste -sd';' -)
  if [ -z "$listed" ]; then
    record FAIL devices "the runtime lists no $prefix device, so models would run on the CPU"
    return 1
  fi
  record PASS devices "$listed"
}

write_config() {
  local port detail
  port=$(python3 "$STEPS" free-port) || { record FAIL config "no free port"; return 1; }
  ENGINE_URL=http://127.0.0.1:$port
  detail=$(python3 "$STEPS" write-config --out "$WORK/engine.json" --work "$WORK" --port "$port" --backend "$BACKEND" \
    --runner "$RUNNER" ${LIBRARIES:+--library "$LIBRARIES"}) || { record FAIL config "$detail"; return 1; }
  record PASS config "$detail"
}

start_engine() {
  local detail chosen
  (cd "$REPO" && exec python3 -m arty_engine --config "$WORK/engine.json" --verbose) >"$WORK/engine.log" 2>&1 &
  ENGINE_PID=$!
  detail=$(python3 "$STEPS" wait --url "$ENGINE_URL" --seconds $ENGINE_START_SECONDS) || {
    record FAIL start "$detail; last log lines: $(tail -5 "$WORK/engine.log" | paste -sd'|' -)"
    return 1
  }
  chosen=$(grep -m1 -o 'GPU backend .*' "$WORK/engine.log")
  case "$chosen" in
    "GPU backend $BACKEND "*) record PASS start "$detail; $chosen" ;;
    *) record FAIL start "$detail; expected backend $BACKEND, the engine logged: ${chosen:-no backend line}"; return 1 ;;
  esac
}

run_step() {
  local step=$1 detail
  shift
  if detail=$(python3 "$STEPS" "$@" --url "$ENGINE_URL"); then
    record PASS "$step" "$detail"
  else
    record FAIL "$step" "$detail"
    return 1
  fi
}

check_models() {
  local vision_ready=true chat_ready=true embed_ready=true
  run_step pull-chat pull --model "$CHAT_MODEL" || chat_ready=false
  run_step pull-embed pull --model "$EMBED_MODEL" || embed_ready=false
  run_step pull-vision pull --model "$VISION_MODEL" || vision_ready=false
  if $chat_ready; then
    run_step chat chat --model "$CHAT_MODEL"
    run_step chat-cpu-only chat --model "$CHAT_MODEL" --cpu-only
  fi
  $embed_ready && run_step embed embed --model "$EMBED_MODEL"
  $vision_ready && run_step picture picture --model "$VISION_MODEL" --image-out "$WORK/rock-42.png"
  run_step loaded loaded
  return 0
}

record_memory() {
  local layers
  record INFO memory "$(python3 "$STEPS" card-memory --backend "$BACKEND")"
  layers=$(grep -ho 'offloaded [0-9]*/[0-9]* layers to GPU' "$WORK"/state/logs/*.log 2>/dev/null | sort -u | paste -sd';' -)
  record INFO gpu-layers "${layers:-no offload line in the runtime logs (none when the engine was started outside this check)}"
}

first_line_or() {
  local fallback=$1 line
  shift
  line=$("$@" 2>/dev/null | head -1)
  echo "${line:-$fallback}"
}

versions() {
  echo "  kernel: $(uname -r)"
  echo "  os: $(first_line_or unknown sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '"')"
  echo "  amdgpu driver: $(first_line_or 'in the kernel (no module version file)' cat /sys/module/amdgpu/version)"
  echo "  python: $(python3 --version 2>&1)"
  echo "  engine: $(first_line_or 'not a git checkout' git -C "$REPO" rev-parse --short HEAD)"
  echo "  runtime: release $RELEASE ${ASSET:-(not downloaded)}"
  echo "  ROCm libraries: $ROCM_LIBRARIES"
  echo "  amd-smi: $(first_line_or 'not installed' amd-smi version)"
  echo "  rocminfo: $(first_line_or 'not installed' command -v rocminfo)"
  echo "  vulkaninfo: $(vulkaninfo --summary 2>/dev/null | grep -m1 'Vulkan Instance Version' | sed 's/.*: *//')"
}

write_report() {
  local overall=PASS line
  case " ${RESULTS[*]} " in *" CHECK "*) overall="PASS, with CHECK lines to compare by hand" ;; esac
  $FAILED && overall=FAIL
  {
    echo "ArtyEngine AMD check, $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo "result: $overall"
    echo "versions:"
    versions
    echo "steps:"
    for line in "${RESULTS[@]}"; do echo "  $line"; done
  } >"$REPORT"
  echo "report: $REPORT ($overall)"
}

stop_engine() {
  local waited=0
  [ -n "$ENGINE_PID" ] || return 0
  kill -TERM "$ENGINE_PID" 2>/dev/null
  while kill -0 "$ENGINE_PID" 2>/dev/null && [ "$waited" -lt "$STOP_SECONDS" ]; do
    sleep 1
    waited=$((waited + 1))
  done
  kill -KILL "$ENGINE_PID" 2>/dev/null
  [ -z "$WORK" ] || pkill -f "$WORK/runtime/" 2>/dev/null
  ENGINE_PID=""
}

clean_up() {
  local status=$1
  trap - EXIT INT TERM
  stop_engine
  if [ -n "$WORK" ]; then
    if $KEEP; then
      record INFO cleanup "engine stopped, kept $WORK"
    elif rm -rf "$WORK"; then
      record PASS cleanup "engine stopped, temporary folder removed"
    else
      record FAIL cleanup "could not remove $WORK"
    fi
  fi
  write_report
  exit "$status"
}

finish() {
  if $FAILED; then clean_up $EXIT_FAILED; fi
  clean_up 0
}

require_tools() {
  local tool missing=""
  for tool in curl python3 sha256sum tar ldd; do
    command -v "$tool" >/dev/null || missing="$missing $tool"
  done
  [ -z "$missing" ] || { record FAIL tools "missing:$missing (see docs/amd-check.md)"; return 1; }
}

main() {
  parse_arguments "$@"
  if $DRY_RUN; then
    print_plan
    exit 0
  fi
  trap 'clean_up $?' EXIT
  trap 'exit 130' INT TERM
  require_tools || exit $EXIT_FAILED
  detect_gpu || exit $EXIT_NO_AMD
  stops_after detect && finish
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/amd-check.XXXXXX") || { record FAIL work "no temporary folder"; exit $EXIT_FAILED; }
  if [ -z "$ENGINE_URL" ]; then
    download_runtime && check_libraries && check_devices || exit $EXIT_FAILED
    stops_after download && finish
    write_config || exit $EXIT_FAILED
    stops_after config && finish
    start_engine || exit $EXIT_FAILED
    stops_after start && finish
  fi
  check_models
  record_memory
  finish
}

main "$@"
