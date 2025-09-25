//
// Created by Meonardo on 11/18/2024.
//

#ifndef ONVIFSINK_H_
#define ONVIFSINK_H_

#include <string>

#include "VideoCommon.h"
#include "utils/MppEncoder.h"

class CLiveAudio;
class CLiveVideo;
namespace rtsp {
class OnvifSink : public RawVideoSink {
 public:
  explicit OnvifSink(std::string_view id, const RtspStreamCfg& cfg);
  ~OnvifSink();

  const std::string& GetId() const {
    return id_;
  }

  // Overrides
  void OnVideoFrameData(const void* buffer, size_t size) override {}
  void OnVideoFrame(const VideoFrameSlot& frame) override;

  // Audio: NOT IMPLEMENT YET!
  void OnAudioBufferData(const void* buffer, uint32_t size);

 private:
  std::string id_;
  RtspStreamCfg cfg_;
  std::unique_ptr<v_enc::MppEncoder> mpp_encoder_;

  // audio
  CLiveAudio* audio_capture_;
  // video
  CLiveVideo* video_capture_;

  void Init();
  void DeInit();
};

}  // namespace rtsp

#endif  // ONVIFSINK_H_
