# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O3 -Wall -s")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("detection")
set(ANDROID_APK OFF)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.

set(RGA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/librga")
set(RKNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/rknpu2")
set(MPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/mpp_new")
set(OSAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/osal")
set(CV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv-4.10.0")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/onvif)

set(SRC_FILES

        src/VideoSink.cpp
        src/OnvifCore.cpp
        src/OnvifSink.cpp
        src/RtspSource.cpp
        src/CameraSource.cpp

        src/utils/DmaAlloc.cpp
        src/utils/MppDecoder.cpp
        src/utils/MppEncoder.cpp
        src/utils/RgaBufferPool.cpp
)

if (ANDROID_APK)
    add_library(${CMAKE_PROJECT_NAME}
            SHARED

            ${SRC_FILES}
    )
    target_compile_definitions(
            ${CMAKE_PROJECT_NAME}

            PRIVATE
            ANDROID_APK
    )
else ()
    if (DET_TYPE STREQUAL "ACTION")
        list(APPEND SRC_FILES src/ActionDetection.cpp)
    else ()
        list(
                APPEND 
                SRC_FILES 
                src/SignDetection.cpp
                src/DetSource.cpp
        )
    endif ()

    add_executable(
            ${CMAKE_PROJECT_NAME}
            ${SRC_FILES}
    )
endif ()

target_compile_definitions(
        ${CMAKE_PROJECT_NAME}

        PRIVATE
        USE_MPP_BUFFER
        USE_QUANTIZED_MODEL
)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif ()

target_link_libraries(${CMAKE_PROJECT_NAME}
        PRIVATE

        log
        m

        # external libraries
        ${RGA_ROOT}/Android/${ANDROID_ABI}/librga.a
        ${RKNN_ROOT}/Android/${ANDROID_ABI}/librknnrt.so
        ${MPP_ROOT}/lib/librockchip_mpp.so
        ${OSAL_ROOT}/lib/libosal.a
        ${CV_ROOT}/libs/libopencv_core.a
        ${CV_ROOT}/libs/libopencv_imgproc.a
        ${CV_ROOT}/libs/libopencv_features2d.a
        ${CV_ROOT}/libs/libopencv_video.a
        ${CV_ROOT}/libs/libopencv_photo.a
        ${CV_ROOT}/libs/libopencv_highgui.a

        # ONVIF
        onvif
)

target_include_directories(${CMAKE_PROJECT_NAME}
        PRIVATE

        ${RGA_ROOT}/include
        ${RKNN_ROOT}/include
        ${MPP_ROOT}/include/rockchip
        ${OSAL_ROOT}/include

        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${CV_ROOT}/include

        # onvif
        ${CMAKE_CURRENT_SOURCE_DIR}/src/onvif/onvif/bm
        ${CMAKE_CURRENT_SOURCE_DIR}/src/onvif/onvif/onvif
        ${CMAKE_CURRENT_SOURCE_DIR}/src/onvif/onvif/http
        ${CMAKE_CURRENT_SOURCE_DIR}/src/onvif/onvif/rtp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/onvif/onvif/rtsp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/onvif/onvif/media
)