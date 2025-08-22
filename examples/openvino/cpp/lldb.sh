#!/usr/bin/env bash
set -euo pipefail

# Start lldb-server in PLATFORM MODE on an Android device, forward ONE port,
# and (optionally) copy your local binary into the device working directory.
#
# - No host->device copy of lldb-server (it's already on the device).
# - Single platform port (no explicit gdbserver port).
# - Optional: push your local ELF and/or a dir of .so deps.
# - Robust 'su' handling WITHOUT '-c' (tries: su 0 / su root / shell).
#
# Typical VS Code (CodeLLDB) flow afterward:
#   initCommands:
#     - platform select remote-android
#     - platform connect connect://localhost:<PLATFORM_PORT>
#   then target/process commands to launch/attach.

# -----------------------------
# Defaults / Config
# -----------------------------
SERIAL=""
REMOTE_LLDB="/data/local/tmp/lldb-server"   # device path to existing lldb-server
PLATFORM_PORT=5432                          # single platform listener port
WORKDIR="/data/local/tmp"                   # where we'll push your bin by default

# Optional copies
BIN_PATH=""         # local path to your ELF to push (optional)
REMOTE_BIN=""       # device path for the pushed ELF; default $WORKDIR/$(basename BIN_PATH)
PUSH_LIB_DIR=""     # local dir of .so deps to push; goes to /data/local/tmp/lldb-standalone/lib

USE_ROOT=0
VERBOSE=0

# Device workspace (for logs/libs; independent of WORKDIR)
R_BASE="/data/local/tmp/lldb-standalone"
R_LIB="${R_BASE}/lib"
R_LOG="${R_BASE}/logs"
R_STARTER="${R_BASE}/start_platform.sh"

usage() {
cat <<EOF
Usage:
  $0 [options]

Options:
  --remote-lldb <path>   Device path to lldb-server (default: /data/local/tmp/lldb-server)
  --platform-port <n>    Platform server port to listen/forward (default: 5432)
  --workdir <dir>        Device working directory (default: /data/local/tmp)
  --bin <path>           LOCAL path to your ELF to push to the device
  --remote-bin <path>    Device path for the pushed ELF (defaults to: \$WORKDIR/\$(basename --bin))
  --libdir <dir>         LOCAL dir of .so deps to push (to ${R_LIB})
  --root                 Try to start lldb-server with su (su 0 → su root → shell)
  -s, --serial <id>      adb -s <serial>
  -v, --verbose          Verbose logs
  -h, --help             This help

Examples:
  # Start platform server and push ./out/yourbin to /data/local/tmp/yourbin
  $0 --bin ./out/yourbin

  # Custom working dir + custom remote bin path
  $0 --workdir /data/local/tmp/myapp --bin ./out/yourbin --remote-bin /data/local/tmp/myapp/runme

  # Also push a local lib dir (useful for LD_LIBRARY_PATH later)
  $0 --bin ./out/yourbin --libdir ./out/lib
EOF
}

log(){ echo "[$(date +%H:%M:%S)] $*"; }
vlog(){ [[ $VERBOSE -eq 1 ]] && echo "[$(date +%H:%M:%S)] $*"; }

# -----------------------------
# Parse args
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-lldb) REMOTE_LLDB="$2"; shift 2;;
    --platform-port) PLATFORM_PORT="$2"; shift 2;;
    --workdir) WORKDIR="$2"; shift 2;;
    --bin) BIN_PATH="$2"; shift 2;;
    --remote-bin) REMOTE_BIN="$2"; shift 2;;
    --libdir) PUSH_LIB_DIR="$2"; shift 2;;
    --root) USE_ROOT=1; shift;;
    -s|--serial) SERIAL="$2"; shift 2;;
    -v|--verbose) VERBOSE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

ADB=(adb)
[[ -n "$SERIAL" ]] && ADB+=(-s "$SERIAL")

# -----------------------------
# Device checks & setup
# -----------------------------
log "Checking device..."
"${ADB[@]}" get-state >/dev/null

log "Using device lldb-server at: ${REMOTE_LLDB}"
"${ADB[@]}" shell "[ -x '${REMOTE_LLDB}' ]" || { echo "Error: ${REMOTE_LLDB} not found or not executable"; exit 1; }

# Ensure workdir & log/lib dirs exist
log "Ensuring device dirs ..."
"${ADB[@]}" shell "mkdir -p '${WORKDIR}' '${R_LIB}' '${R_LOG}'"

# -----------------------------
# Optional: push local libs
# -----------------------------
if [[ -n "${PUSH_LIB_DIR}" ]]; then
  log "Pushing .so from ${PUSH_LIB_DIR} -> ${R_LIB} ..."
  "${ADB[@]}" shell "rm -rf '${R_LIB}' && mkdir -p '${R_LIB}'"
  "${ADB[@]}" push "${PUSH_LIB_DIR}/." "${R_LIB}" >/dev/null
fi

# -----------------------------
# Optional: push local binary
# -----------------------------
if [[ -n "${BIN_PATH}" ]]; then
  if [[ -z "${REMOTE_BIN}" ]]; then
    BN="$(basename "${BIN_PATH}")"
    REMOTE_BIN="${WORKDIR}/${BN}"
  fi
  log "Pushing binary ${BIN_PATH} -> ${REMOTE_BIN} ..."
  "${ADB[@]}" shell "mkdir -p '$(dirname "${REMOTE_BIN}")'"
  "${ADB[@]}" push "${BIN_PATH}" "${REMOTE_BIN}" >/dev/null
  "${ADB[@]}" shell "chmod 755 '${REMOTE_BIN}'"
  log "Binary pushed. Remote path: ${REMOTE_BIN}"
fi

# -----------------------------
# Create starter (device): start lldb-server platform in background
# -----------------------------
TMP_START="$(mktemp)"
cat > "${TMP_START}" <<'EOS'
#!/system/bin/sh
LLDB_SRV="$1"; PLAT_PORT="$2"; LOGDIR="$3"
mkdir -p "$LOGDIR" 2>/dev/null

CMD="$LLDB_SRV platform --server --listen *:$PLAT_PORT \
  --log-file $LOGDIR/lldb-platform.log --log-channels 'lldb host platform'"

# kill the lldb-server before running a new debug session
pkill lldb-server
if command -v nohup >/dev/null 2>&1; then
  nohup sh -c "$CMD" </dev/null >>"$LOGDIR/platform-stdout.log" 2>&1 &
else
  sh -c "$CMD" </dev/null >>"$LOGDIR/platform-stdout.log" 2>&1 &
fi
EOS

log "Pushing starter script..."
"${ADB[@]}" push "${TMP_START}" "${R_STARTER}" >/dev/null
"${ADB[@]}" shell "chmod 755 '${R_STARTER}'"
rm -f "${TMP_START}"

# -----------------------------
# Port forward (host localhost -> device)
# -----------------------------
log "Forwarding tcp:${PLATFORM_PORT} -> device tcp:${PLATFORM_PORT}"
"${ADB[@]}" forward --remove "tcp:${PLATFORM_PORT}" >/dev/null 2>&1 || true
"${ADB[@]}" forward "tcp:${PLATFORM_PORT}" "tcp:${PLATFORM_PORT}"

# -----------------------------
# Start platform server (robust root flow, no '-c')
# -----------------------------
if [[ ${USE_ROOT} -eq 1 ]]; then
  log "Starting lldb-server (platform mode) with root ..."
  if "${ADB[@]}" shell su 0 /system/bin/sh "${R_STARTER}" "${REMOTE_LLDB}" "${PLATFORM_PORT}" "${R_LOG}"; then
    :
  elif "${ADB[@]}" shell su root /system/bin/sh "${R_STARTER}" "${REMOTE_LLDB}" "${PLATFORM_PORT}" "${R_LOG}"; then
    :
  else
    log "Root start failed; falling back to shell user."
    "${ADB[@]}" shell /system/bin/sh "${R_STARTER}" "${REMOTE_LLDB}" "${PLATFORM_PORT}" "${R_LOG}"
  fi
else
  log "Starting lldb-server (platform mode) as shell user ..."
  "${ADB[@]}" shell /system/bin/sh "${R_STARTER}" "${REMOTE_LLDB}" "${PLATFORM_PORT}" "${R_LOG}"
fi

log "Ready.

===============================================
 Platform server is up
-----------------------------------------------
Platform port : ${PLATFORM_PORT}   (adb forward: localhost:${PLATFORM_PORT})
Working dir   : ${WORKDIR}
Remote bin    : ${REMOTE_BIN:-<not pushed in this run>}

Device logs   :
  ${R_LOG}/lldb-platform.log
  ${R_LOG}/platform-stdout.log

Next steps in VS Code (CodeLLDB):
  initCommands:
    - platform select remote-android
    - platform connect connect://localhost:${PLATFORM_PORT}
  Then either:
    - target create ${REMOTE_BIN:-/data/local/tmp/yourbin}
    - process launch -- <args>
    OR
    - process attach -p <PID>
==============================================="
