#ifndef RTSP_SOURCE_H_
#define RTSP_SOURCE_H_

#include <string>

// For RTSP client
#include "rtsp_cln.h"
#include "VideoCommon.h"

#include "utils/MppDecoder.h"

namespace rtsp {
class RtspSource : public VideoSource {
 public:
  explicit RtspSource(std::string_view id, std::string_view url);
  ~RtspSource() override;

  // Overrides
  void AddVideoSink(VideoSink* sink) override;
  void RemoveVideoSink(VideoSink* sink) override;
  [[nodiscard]] std::string GetId() const override { return id_; }
  VideoFrameSlot& GetVideoFrame() override {
    return current_frame_;
  }

  // Actions
  bool Start();
  void Stop();

 private:
  std::string id_;
  std::string url_;
  CRtspClient* rtsp_client_;

  // video sinks
  std::vector<VideoSink*> sinks_;
  std::mutex sink_mutex_;

  // decoder
  v_dec::MppDecoder* mpp_decoder_;

  // video original resolution
  RK_U32 video_width_;
  RK_U32 video_height_;
  VideoFrameSlot current_frame_;

  void InitRtspClient();
  // RTSP client callbacks
  int RtspNotifyCallback(int event);
  int RtspVideoCallback(uint8* data, int len, uint32 ts, uint16 seq);

  // Decode callback
  void OnDecodedFrame(const VideoFrameSlot& frame);
};
}  // namespace rtsp

#endif  // RTSP_SOURCE_H_