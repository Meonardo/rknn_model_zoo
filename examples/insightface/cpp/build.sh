#!/bin/bash

# Usage:
# ./build.sh arm64-v8a Debug

# get target from command line argument
# target are: arm64-v8a, armeabi-v7a
TARGET=$1
if [ $# -ne 1 ]; then
    echo "Usage: $0 <target>"
    echo "No target specified, use default arm64-v8a"
    TARGET="arm64-v8a"
fi
echo "Target: $TARGET"

# get build mode from command line input
# mode are: Debug, Release
MODE=$2
if [ -z "$MODE" ]; then
    echo "No build mode specified, use default Release"
    MODE="Release"
fi
echo "Build mode: $MODE"

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
echo "Current dir: $SCRIPT_DIR"

# Create the build directory & install directory
rm -fr $SCRIPT_DIR/build
mkdir -p $SCRIPT_DIR/build

INSTALL_DIR=$SCRIPT_DIR/install/$TARGET/$MODE
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/lib
mkdir -p $INSTALL_DIR/bin

ANDROID_NDK=/home/hd/project/Android/NDK/android-ndk-r25c
# set the toolchain.cmake
TOOLCHAIN=$ANDROID_NDK/build/cmake/android.toolchain.cmake
ANDROID_ABI=$TARGET
ANDROID_PLATFORM=android-29
STRIP=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip

# Run CMake
cmake -B $SCRIPT_DIR/build -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN -DANDROID_ABI=$ANDROID_ABI -DANDROID_PLATFORM=$ANDROID_PLATFORM -DCMAKE_BUILD_TYPE=$MODE

# Build the project
cd $SCRIPT_DIR/build
make -j$(nproc)

if [ "$MODE" == "Release" ]; then
  # Strip the binary
  $STRIP --strip-unneeded insightface
fi

# Copy the binary to the install directory
cp insightface $INSTALL_DIR/bin
# Copy deps lib to the install directory
cp ../../../../3rdparty/rknpu2/Android/arm64-v8a/librknnrt.so $INSTALL_DIR/lib
cp ../deps/mpp_new/lib/librockchip_mpp.so $INSTALL_DIR/lib
cp ../deps/mpp_new/lib/librockchip_vpu.so $INSTALL_DIR/lib

echo "All done!"