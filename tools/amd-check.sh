#!/bin/bash
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
STEPS=$REPO/tools/amd_check_steps.py
RELEASE=runtime-b12110
RELEASE_API=https://api.github.com/repos/drew-codes-things/ArtyEngine/releases/tags/$RELEASE
ASSET_URL_PREFIX=https://github.com/drew-codes-things/ArtyEngine/releases/download/$RELEASE/arty-runtime-
ROCM_ASSET_KIND=linux-rocm-10.0-x64
VULKAN_ASSET_KIND=linux-vulkan-x64
ROCM_WHEEL_INDEX=https://stable.repo.amd.com/rocm/whl-next/
ROCM_WHEEL_VERSION=10.0.0
ROCM_LIBRARY_FILES="libamdhip64.so.7 libhipblas.so.3 librocblas.so.5"
ROCM_BUILD_TARGETS="gfx908 gfx90a gfx942 gfx950 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1200 gfx1201"
CHAT_MODEL=hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0
EMBED_MODEL=nomic-embed-text
VISION_MODEL=hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M
DRM=${AMD_CHECK_DRM:-/sys/class/drm}
DEV=${AMD_CHECK_DEV:-/dev}
KFD=${AMD_CHECK_KFD:-/sys/class/kfd/kfd/topology/nodes}
WORK_PARENT=${TMPDIR:-/var/tmp}
ENGINE_START_SECONDS=60
STOP_SECONDS=30
EXIT_FAILED=1
EXIT_NO_AMD=2
EXIT_USAGE=64
EXIT_INTERRUPTED=130
STEP_ORDER="detect download config start"

DRY_RUN=false
KEEP=false
INSTALL_ROCM_WHEELS=true
BACKEND=""
STOP_AFTER=""
ENGINE_URL=""
REPORT=$PWD/amd-check-report.txt
WORK=""
LOGS=""
ENGINE_PID=""
ASSET=""
BUILD=""
RUNNER=""
LIBRARIES=""
GFX=""
GPU_SUMMARY=""
ROCM_LIBRARIES="not used"
FAILED=false
INTERRUPTED=false
REACHED_END=false
NOT_CHECKED=""
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
  echo "$ASSET_URL_PREFIX$1.zip"
}

print_plan() {
  cat <<EOF
AMD check plan (dry run: nothing is downloaded, started or written)
  1. detect the GPU: lspci, $DEV/kfd, amdgpu render nodes in $DRM, the card's gfx target (rocminfo, else $KFD),
     vulkaninfo when installed; ROCm when $DEV/kfd is there and the ROCm build carries code for that target, else Vulkan
  2. download release $RELEASE of the model runtime into a temporary folder under $WORK_PARENT and check its sha256
     against the release listing (no listing, no run):
$(asset_url $ROCM_ASSET_KIND)
$(asset_url $VULKAN_ASSET_KIND)
     ROCm only: when the system has no ROCm $ROCM_WHEEL_VERSION libraries, install AMD's
     rocm[libraries,device-<gfx target>]==$ROCM_WHEEL_VERSION packages from $ROCM_WHEEL_INDEX into that folder, as your
     own user; then ask the runtime which devices it sees
  3. write a temporary engine config listening on a free 127.0.0.1 port
  4. start the engine from $REPO
  5. pull through the engine, from Hugging Face: $CHAT_MODEL, $EMBED_MODEL, $VISION_MODEL
  6. chat check (expects Paris), the same chat on the CPU (num_gpu 0), embed check (768 numbers),
     picture check on a generated picture of the text ROCK 42; after each, the runtime's log must show layers on the
     card (none for the CPU chat)
  7. record the free card memory the engine reads
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

in_list() {
  case " $2 " in *" $1 "*) return 0 ;; esac
  return 1
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

gfx_name() {
  printf 'gfx%d%x%x' "$1" "$2" "$3"
}

override_target() {
  local major minor stepping
  [[ "${HSA_OVERRIDE_GFX_VERSION:-}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || return 1
  major=${BASH_REMATCH[1]} minor=${BASH_REMATCH[2]} stepping=${BASH_REMATCH[3]}
  gfx_name "$major" "$minor" "$stepping"
}

sysfs_target() {
  local version
  version=$(awk '$1 == "gfx_target_version" && $2 > 0 { print $2; exit }' "$KFD"/*/properties 2>/dev/null)
  [[ "$version" =~ ^[0-9]+$ ]] || return 1
  gfx_name $((version / 10000)) $((version / 100 % 100)) $((version % 100))
}

card_target() {
  override_target && return 0
  rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' && return 0
  sysfs_target
}

detect_gpu() {
  local pci render kfd=no vulkan_device why=""
  pci=$(amd_pci_lines)
  render=$(amd_render_node) || render=""
  [ -e "$DEV/kfd" ] && kfd=yes
  GFX=$(card_target) || GFX=""
  vulkan_device=$(vulkaninfo --summary 2>/dev/null | grep -m1 'deviceName' | sed 's/.*= *//' || true)
  GPU_SUMMARY="lspci: ${pci:-no AMD display device}; render node: ${render:-none}; /dev/kfd: $kfd; card target: ${GFX:-not readable}${HSA_OVERRIDE_GFX_VERSION:+ (from HSA_OVERRIDE_GFX_VERSION)}; vulkaninfo: ${vulkan_device:-not installed or no device}"
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
    BACKEND=vulkan
    if [ $kfd = no ]; then
      why=" (no /dev/kfd)"
    elif [ -n "$GFX" ] && in_list "$GFX" "$ROCM_BUILD_TARGETS"; then
      BACKEND=rocm
    else
      why=" (the ROCm build carries no code for card target ${GFX:-not readable})"
    fi
  fi
  record PASS detect "backend $BACKEND$why. $GPU_SUMMARY"
}

published_digest() {
  curl -fsSL --retry 3 -o "$WORK/release.json" "$RELEASE_API" 2>/dev/null || return 1
  python3 "$STEPS" published-digest "$WORK/release.json" "$ASSET"
}

pinned_digest() {
  if [ -n "${AMD_CHECK_RUNTIME_SHA256:-}" ]; then
    echo "$AMD_CHECK_RUNTIME_SHA256"
  else
    python3 "$STEPS" pinned-digest "$1"
  fi
}

download_runtime() {
  local kind=$ROCM_ASSET_KIND url published actual pinned second_check
  [ "$BACKEND" = vulkan ] && kind=$VULKAN_ASSET_KIND
  url=$(asset_url "$kind")
  ASSET=${url##*/}
  BUILD="release $RELEASE, $kind build"
  curl -fL --retry 3 -sS -o "$WORK/runtime.zip" "$url" || { record FAIL download "could not download $url"; return 1; }
  actual=$(sha256sum "$WORK/runtime.zip" | cut -d' ' -f1)
  pinned=$(pinned_digest "$kind")
  if [ -n "$pinned" ] && [ "$actual" != "$pinned" ]; then
    record FAIL download "$BUILD has sha256 $actual, but this kit pins $pinned, so it will not be run"
    return 1
  fi
  if ! published=$(published_digest); then
    if [ -z "$pinned" ]; then
      record FAIL download "this kit pins no digest for the $BUILD and the release listing could not be read, so it will not be run"
      return 1
    fi
    second_check="the release listing could not be read as a second check"
  elif [ "$actual" != "$published" ]; then
    record FAIL download "$BUILD has sha256 $actual, but the release lists $published"
    return 1
  else
    second_check="the release listing agrees"
  fi
  mkdir -p "$WORK/runtime"
  python3 "$STEPS" unpack "$WORK/runtime.zip" "$WORK/runtime" >/dev/null || { record FAIL download "could not unpack the $BUILD"; return 1; }
  RUNNER=$(find "$WORK/runtime" -maxdepth 1 -type f -name 'arty-runtime-server' | head -1)
  [ -n "$RUNNER" ] || { record FAIL download "no arty-runtime-server in the $BUILD"; return 1; }
  if [ -n "$pinned" ]; then
    record PASS download "$BUILD, sha256 $actual matches the pinned digest ($second_check)"
  else
    record PASS download "$BUILD, sha256 $actual matches the release listing (this kit pins no digest for it yet)"
  fi
}

missing_libraries() {
  LD_LIBRARY_PATH="$1" ldd "$2" 2>&1 | grep 'not found' | awk '{print $1}' | paste -sd' ' -
}

make_rocm_venv() {
  [ "$BACKEND" = rocm ] && $INSTALL_ROCM_WHEELS || return 0
  python3 -m venv "$WORK/rocm-libs" >/dev/null 2>&1 && return 0
  record FAIL tools "python3 -m venv does not work, and AMD's ROCm packages need it: install the python3-venv package (see docs/amd-check.md), or run with --no-rocm-wheels on a system with ROCm $ROCM_WHEEL_VERSION"
  return 1
}

rocm_library_folders() {
  local name found folder folders=""
  for name in $ROCM_LIBRARY_FILES; do
    found=$(find "$WORK/rocm-libs" -name "$name*" | head -1)
    [ -n "$found" ] || continue
    folder=$(dirname "$found")
    in_list "$folder" "${folders//:/ }" || folders=${folders:+$folders:}$folder
  done
  echo "$folders"
}

installed_device_packages() {
  find "$WORK/rocm-libs" -maxdepth 5 -type d -name 'rocm_sdk_device_*' 2>/dev/null | sed 's/.*rocm_sdk_device_/rocm-sdk-device-/; s/-[0-9].*//' | sort -u | paste -sd' ' -
}

install_rocm_wheels() {
  local package
  if [ -z "$GFX" ] || ! in_list "$GFX" "$ROCM_BUILD_TARGETS"; then
    WHEEL_PROBLEM="AMD has no kernel package for card target ${GFX:-not readable} that the ROCm build can use"
    return 1
  fi
  package="rocm[libraries,device-$GFX]==$ROCM_WHEEL_VERSION"
  echo "installing AMD's $package into $WORK/rocm-libs (a large download)"
  if ! PIP_CACHE_DIR="$WORK/pip-cache" "$WORK/rocm-libs/bin/pip" install -q --index-url "$ROCM_WHEEL_INDEX" "$package"; then
    WHEEL_PROBLEM="AMD's ROCm packages did not install"
    return 1
  fi
  LIBRARIES=$(rocm_library_folders)
  ROCM_LIBRARIES="AMD packages $package, installed: $(installed_device_packages)"
}

check_libraries() {
  local backend_library missing
  WHEEL_PROBLEM=""
  LIBRARIES=""
  [ "$BACKEND" = rocm ] && ROCM_LIBRARIES="system"
  backend_library=$(find "$WORK/runtime" -maxdepth 1 -name "lib*-${BACKEND/rocm/hip}.so" | head -1)
  [ -n "$backend_library" ] || { record FAIL libraries "no $BACKEND backend library in the $BUILD"; return 1; }
  missing=$(missing_libraries "$WORK/runtime" "$backend_library")
  if [ -n "$missing" ] && [ "$BACKEND" = rocm ] && $INSTALL_ROCM_WHEELS; then
    install_rocm_wheels || { record FAIL libraries "missing $missing, and $WHEEL_PROBLEM"; return 1; }
    missing=$(missing_libraries "$LIBRARIES:$WORK/runtime" "$backend_library")
  fi
  if [ -n "$missing" ]; then
    record FAIL libraries "the $BACKEND build cannot load: $missing (see docs/amd-check.md)"
    return 1
  fi
  record PASS libraries "every library of the $BACKEND backend loads (ROCm libraries: $ROCM_LIBRARIES${LIBRARIES:+, from $LIBRARIES})"
}

check_devices() {
  local listed prefix=ROCm amd_listed
  [ "$BACKEND" = vulkan ] && prefix=Vulkan
  listed=$(LD_LIBRARY_PATH="${LIBRARIES:+$LIBRARIES:}$WORK/runtime" "$RUNNER" --list-devices 2>&1 | grep -E "^ *${prefix}[0-9]+:")
  if [ -z "$listed" ]; then
    record FAIL devices "the runtime lists no $prefix device, so models would run on the CPU"
    return 1
  fi
  amd_listed=$listed
  [ "$BACKEND" = vulkan ] && amd_listed=$(grep -Ei 'amd|radeon|radv' <<<"$listed")
  if [ -z "$amd_listed" ]; then
    record FAIL devices "the runtime lists only other vendors' Vulkan devices ($(paste -sd';' - <<<"$listed")), so the check would not test the AMD card"
    return 1
  fi
  record PASS devices "$(paste -sd';' - <<<"$amd_listed")"
}

write_config() {
  local port detail folder library_arguments=()
  port=$(python3 "$STEPS" free-port) || { record FAIL config "no free port"; return 1; }
  ENGINE_URL=http://127.0.0.1:$port
  for folder in ${LIBRARIES//:/ }; do
    library_arguments+=(--library "$folder")
  done
  detail=$(python3 "$STEPS" write-config --out "$WORK/engine.json" --work "$WORK" --port "$port" --backend "$BACKEND" \
    --runner "$RUNNER" "${library_arguments[@]}") || { record FAIL config "$detail"; return 1; }
  record PASS config "$detail"
}

start_engine() {
  local detail chosen
  (cd "$REPO" && XDG_CACHE_HOME="$WORK/cache" exec python3 -m arty_engine --config "$WORK/engine.json" --verbose) >"$WORK/engine.log" 2>&1 &
  ENGINE_PID=$!
  LOGS=$WORK/state/logs
  detail=$(python3 "$STEPS" wait --url "$ENGINE_URL" --seconds $ENGINE_START_SECONDS) || {
    record FAIL start "$detail; last log lines: $(tail -5 "$WORK/engine.log" | paste -sd'|' -)"
    return 1
  }
  chosen=$(grep -m1 -o 'GPU backend .*' "$WORK/engine.log" | sed 's/model runner [^,]*/model runner from the download/')
  case "$chosen" in
    "GPU backend $BACKEND "*"card memory not readable"*)
      record FAIL start "$detail; $chosen: the engine could read the card's memory with none of amd-smi, rocm-smi or sysfs"
      return 1 ;;
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

check_offload() {
  local step=$1 expect=$2 detail
  if [ -z "$LOGS" ]; then
    NOT_CHECKED="GPU offload (the engine was started outside this check, so its runtime logs are not here)"
    return 0
  fi
  if detail=$(python3 "$STEPS" offload --logs "$LOGS" --since "$WORK/log-sizes.json" --expect "$expect"); then
    record PASS "offload-$step" "$detail"
  else
    record FAIL "offload-$step" "$detail"
  fi
}

run_model_step() {
  local step=$1 expect=$2
  shift 2
  [ -z "$LOGS" ] || python3 "$STEPS" log-sizes --logs "$LOGS" >"$WORK/log-sizes.json"
  run_step "$step" "$@" || return 1
  check_offload "$step" "$expect"
}

check_models() {
  local vision_ready=true chat_ready=true embed_ready=true
  run_step pull-chat pull --model "$CHAT_MODEL" || chat_ready=false
  run_step pull-embed pull --model "$EMBED_MODEL" || embed_ready=false
  run_step pull-vision pull --model "$VISION_MODEL" || vision_ready=false
  if $chat_ready; then
    run_model_step chat card chat --model "$CHAT_MODEL"
    run_model_step chat-cpu-only cpu chat --model "$CHAT_MODEL" --cpu-only
  fi
  $embed_ready && run_model_step embed card embed --model "$EMBED_MODEL"
  $vision_ready && run_model_step picture card picture --model "$VISION_MODEL" --image-out "$WORK/rock-42.png"
  run_step loaded loaded
  return 0
}

record_memory() {
  record INFO memory "$(python3 "$STEPS" card-memory --backend "$BACKEND")"
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
  echo "  runtime: ${BUILD:-release $RELEASE (not downloaded)}"
  echo "  ROCm libraries: $ROCM_LIBRARIES"
  echo "  amd-smi: $(first_line_or 'not installed' amd-smi version)"
  echo "  rocminfo: $(first_line_or 'not installed' command -v rocminfo)"
  echo "  vulkaninfo: $(vulkaninfo --summary 2>/dev/null | grep -m1 'Vulkan Instance Version' | sed 's/.*: *//')"
}

overall_result() {
  if $FAILED; then
    echo FAIL
  elif $INTERRUPTED; then
    echo "INCOMPLETE (interrupted; only the steps below ran)"
  elif ! $REACHED_END; then
    echo "INCOMPLETE (the check ended before its last step)"
  elif [ -n "$STOP_AFTER" ]; then
    echo "INCOMPLETE (stopped after $STOP_AFTER as asked; every step that ran passed)"
  elif [ -n "$NOT_CHECKED" ]; then
    echo "INCOMPLETE (not checked: $NOT_CHECKED)"
  else
    echo PASS
  fi
}

write_report() {
  local overall line
  overall=$(overall_result)
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
  REACHED_END=true
  if $FAILED; then clean_up $EXIT_FAILED; fi
  clean_up 0
}

require_tools() {
  local tool missing=""
  for tool in curl python3 sha256sum ldd; do
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
  trap 'INTERRUPTED=true; exit $EXIT_INTERRUPTED' INT TERM
  require_tools || exit $EXIT_FAILED
  detect_gpu || exit $EXIT_NO_AMD
  stops_after detect && finish
  WORK=$(mktemp -d "$WORK_PARENT/amd-check.XXXXXX") || { record FAIL work "no temporary folder in $WORK_PARENT"; exit $EXIT_FAILED; }
  if [ -z "$ENGINE_URL" ]; then
    make_rocm_venv || exit $EXIT_FAILED
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
