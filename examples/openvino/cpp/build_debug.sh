set -e

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
echo "Current dir: $SCRIPT_DIR"

cd $SCRIPT_DIR

# 1. run build.sh

./build.sh arm64-v8a Debug

# 2. run lldb.sh
./lldb.sh \
  --bin $SCRIPT_DIR/build/action_detector \
  --libdir $SCRIPT_DIR/install/arm64-v8a/Debug/lib \
  --remote-lldb /data/local/tmp/lldb-server \
  --remote-bin /data/local/tmp/lldb-standalone/bin/action_detector \
  --root \
  -v