//
// Created by Meonardo on 11/18/2024.
//

#include "OnvifSink.h"

#include <string.h>

////////////////////////////////////////////////////////////////////////
// this must include before any other ONVIF headers
#include "sys_inc.h"
////////////////////////////////////////////////////////////////////////
#include "Common.h"
#include "live_audio.h"
#include "live_video.h"
#include "onvif.h"
#include "onvif_cfg.h"
#include "rtsp_cfg.h"

#define TAG "OnvifSink"

#define AUDIO_CODEC "AAC"
#define DEFAULT_AUDIO_SAMPLE_RATE 8000
#define DEFAULT_AUDIO_CHANNELS 1

extern "C" ONVIF_CLS g_onvif_cls;

namespace rtsp {

OnvifSink::OnvifSink(std::string_view id, const RtspStreamCfg& cfg)
    : id_(id), cfg_(cfg), audio_capture_(nullptr), video_capture_(nullptr), mpp_encoder_(nullptr) {
  LOGI(TAG, "Create OnvifSink with id: %s", id_.c_str());

  Init();
}

OnvifSink::~OnvifSink() {
  LOGI(TAG, "Destroying OnvifSink with id: %s", id_.c_str());

  DeInit();

  LOGI(TAG, "OnvifSink with id: %s destroyed", id_.c_str());
}

void OnvifSink::Init() {
  // Create the encoder
  EncodedVideoInfo enc_info = cfg_.video_info;
  mpp_encoder_ = std::make_unique<v_enc::MppEncoder>(enc_info, this);
  // Set the callback to receive encoded data
  mpp_encoder_->SetCallback([](void* userdata, const char* data, size_t size) {
    auto* sink = reinterpret_cast<OnvifSink*>(userdata);
    sink->video_capture_->procData((uint8*) data, (int) size);
  });

  // Set media stream info
  int idx = (int) strtol(id_.c_str(), nullptr, 10);
  idx--;  // match the index in the array of live video instances

  if (idx < 0 || idx >= MAX_LIVE_VIDEO_NUMS) {
    LOGW(TAG, "invalid index: %d", idx);
    // assert(false);
    idx = 4;  // default to the last index
  }

  // get the live video instance
  video_capture_ = CLiveVideo::getInstance(idx);
  video_capture_->setStreamName(id_.c_str());
  // audio_capture_ = CLiveAudio::getInstance(idx);

  if (video_capture_ == nullptr) {
    LOGE(TAG, "failed to get live video or audio instance");
    assert(false);
  }
}

void OnvifSink::DeInit() {
  LOGI(TAG, "DeInit OnvifSink with id: %s", id_.c_str());

  if (video_capture_ != nullptr) {
    video_capture_->freeInstance(video_capture_->m_nStreamIndex);
    video_capture_ = nullptr;
  }

  // if (audio_capture_ != nullptr) {
  //   audio_capture_->freeInstance(audio_capture_->m_nStreamIndex);
  //   audio_capture_ = nullptr;
  // }

  if (mpp_encoder_) {
    mpp_encoder_->Stop();
    mpp_encoder_.reset();
    mpp_encoder_ = nullptr;
  }
}

void OnvifSink::OnAudioBufferData(const void* buffer, uint32_t size) {
  if (audio_capture_ == nullptr) {
    LOGE(TAG, "audio capture is not initialized");
    return;
  }

  // Process the audio data
  audio_capture_->procData((uint8*) buffer, (int) size, DEFAULT_AUDIO_SAMPLE_RATE);
}

void OnvifSink::OnVideoFrame(const VideoFrameSlot& frame) {
  if (mpp_encoder_ == nullptr) {
    LOGE(TAG, "video encoder is not initialized");
    return;
  }

  // Encode video frame
  mpp_encoder_->Encode(frame);
}

}  // namespace rtsp