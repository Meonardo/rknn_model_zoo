//
// Created by Meonardo on 2/5/2024.
//

#ifndef GSTREAMERANDROID_GSTREAMER_SRC_MAIN_CPP_VIDEOSINK_H_
#define GSTREAMERANDROID_GSTREAMER_SRC_MAIN_CPP_VIDEOSINK_H_

#include <string>

#include "mpp_frame.h"

#define MPP_ALIGN(x, a)         (((x)+(a)-1)&~((a)-1))

#define USE_VIRTUAL_ADDRESS 0
#define USE_MPP_BUFFER 1

// global constants
constexpr RK_U32 kMaxDecodedFrameBufferCount = 24;
constexpr int kDefaultRingBufferSize = 4;
constexpr int kDefaultFps = 25;
constexpr int kDefaultWaitingTimeInMs = 1000 / kDefaultFps;

// for base video buffer
// For 1080p
constexpr int kBaseVideoWidth = 1920;
constexpr int kBaseVideoHeight = 1080;

// For 4K
//constexpr int kBaseVideoWidth = 3840;
//constexpr int kBaseVideoHeight = 2160;

#if USE_VIRTUAL_ADDRESS
constexpr int kBaseVerticalStride = kBaseVideoHeight;
#else
constexpr int kBaseVerticalStride = MPP_ALIGN(kBaseVideoHeight, 16);
#endif

struct VideoBuffSlot {
  size_t size;
  RK_U32 eos;
  char *data;
};

struct VideoFrameSlot {
  RK_U32 width;
  RK_U32 height;
  RK_U32 width_stride;
  RK_U32 height_stride;
  MppFrameFormat format;
  int fd;
  bool need_refresh;
  bool eos;
};

class VideoSink {
 public:
  virtual ~VideoSink() = default;

  // raw video buffer
  virtual void OnVideoFrame(const VideoFrameSlot &frame) = 0;
  virtual void OnVideoFrameData(const void *buffer, size_t size) = 0;

  // encoded video buffer
  virtual void OnVideoBufferData(const void *buffer, size_t size, const std::string& source_id) = 0;
};

class EncodedVideoSink : public VideoSink {
 public:
  ~EncodedVideoSink() override = default;

  // override raw video sink part,
  // these methods are not available for who implements this interface
  void OnVideoFrameData(const void *buffer, size_t size) override {}
  void OnVideoFrame(const VideoFrameSlot &frame) override {}
};

class RawVideoSink : public VideoSink {
 public:
  ~RawVideoSink() override = default;

  // override encoded video sink part,
  // these methods are not available for who implements this interface
  void OnVideoBufferData(const void *buffer, size_t size, const std::string& source_id) override {}
};

class VideoSource {
 public:
  virtual ~VideoSource() = default;

  virtual void AddVideoSink(VideoSink *sink) = 0;
  virtual void RemoveVideoSink(VideoSink *sink) = 0;
  [[nodiscard]] virtual std::string GetId() const = 0;
  virtual VideoFrameSlot& GetVideoFrame() = 0;
};

struct EncodedVideoInfo {
  uint32_t width;
  uint32_t height; // video resolution, default is 1920x1080
  uint32_t bitrate; // video bitrate, default is 4000000(for H.264) or 1000000(for H.265)
  uint16_t fps; // frame per second, default is 25
  int codec; // 0: h264, 1: h265
  int rate_control; // 0: VBR, 1: CBR, 2: FixQP, 3: AVBR
  std::string codec_name; // h264, h265
  std::string stream_format; // byte-stream, avc, annexb
  std::string profile_level_id; // h264: 4D0029, h265: 018028
  std::string alignment; // au, nal
  int pixel_format; // 0: yuv420sp(nv12), 1: RGBA, 2: RGB

  EncodedVideoInfo();
  ~EncodedVideoInfo() = default;
};

#endif //GSTREAMERANDROID_GSTREAMER_SRC_MAIN_CPP_VIDEOSINK_H_
