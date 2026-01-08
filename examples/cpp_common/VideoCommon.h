#ifndef VIDEO_COMMON_H_
#define VIDEO_COMMON_H_

#include <string>

#include "VideoSink.h"

namespace rtsp {
  
struct RtspStreamCfg {
  std::string url;
  EncodedVideoInfo video_info;
};

enum class RtspMessageType {
  RTSP_MT_NULL = 0,
  RTSP_MT_DESCRIBE,
  RTSP_MT_ANNOUNCE,
  RTSP_MT_OPTIONS,
  RTSP_MT_PAUSE,
  RTSP_MT_PLAY,
  RTSP_MT_RECORD,
  RTSP_MT_REDIRECT,
  RTSP_MT_SETUP,
  RTSP_MT_SET_PARAMETER,
  RTSP_MT_GET_PARAMETER,
  RTSP_MT_TEARDOWN
};

class RtspClientEventHandler {
 public:
  virtual ~RtspClientEventHandler() = default;
  virtual void OnRtspClientStateChanged(std::string_view client_id, RtspMessageType msg_sub_type,
                                        std::string_view client_ip, uint16_t client_port) = 0;
};
}  // namespace rtsp

#endif  // VIDEO_COMMON_H_