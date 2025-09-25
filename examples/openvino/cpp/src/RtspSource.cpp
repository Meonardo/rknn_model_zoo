#include "RtspSource.h"

#include <Common.h>

#define TAG "RtspSource"

namespace rtsp {
RtspSource::RtspSource(std::string_view id, std::string_view url)
    : id_(id),
      url_(url),
      rtsp_client_(new CRtspClient()),
      mpp_decoder_(
          new v_dec::MppDecoder(MPP_VIDEO_CodingAVC, kBaseVideoWidth, kBaseVideoHeight, this)),
      video_width_(1920),
      video_height_(1080) {
  LOGI(TAG, "Create RtspSource(id=%s) with url: %s", id_.c_str(), url_.c_str());
  InitRtspClient();
  mpp_decoder_->SetCallback([](void* userdata, RK_U32 width_stride, RK_U32 height_stride,
                               RK_U32 width, RK_U32 height, MppFrameFormat format, int fd) {
    auto* source = reinterpret_cast<RtspSource*>(userdata);
    source->OnDecodedFrame({width, height, width_stride, height_stride, format, fd});
  });
}

RtspSource::~RtspSource() {
  LOGI(TAG, "Destroying RtspSource(id=%s) with url: %s", id_.c_str(), url_.c_str());

  // Stop RTSP client and release resources
  if (rtsp_client_) {
    rtsp_client_->rtsp_stop();
    delete rtsp_client_;
    rtsp_client_ = nullptr;
  }

  // Release decoder
  if (mpp_decoder_) {
    mpp_decoder_->PutPacket(nullptr, 0, 1);  // send EOS
    // sleep for a while to wait for the decoder to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    delete mpp_decoder_;
    mpp_decoder_ = nullptr;
  }

  LOGI(TAG, "RtspSource(id=%s) with url: %s destroyed", id_.c_str(), url_.c_str());
}

void RtspSource::InitRtspClient() {
  if (!rtsp_client_) {
    LOGE(TAG, "RTSP client is not initialized");
    return;
  }

  rtsp_client_->set_rtp_over_udp(1);  // Use RTP over UDP
  rtsp_client_->set_notify_cb(
      [](int event, void* userdata) -> int {
        auto source = static_cast<RtspSource*>(userdata);
        return source->RtspNotifyCallback(event);
      },
      this);
  rtsp_client_->set_video_cb(
      [](uint8* data, int len, uint32 ts, uint16 seq, void* userdata) -> int {
        auto source = static_cast<RtspSource*>(userdata);
        return source->RtspVideoCallback(data, len, ts, seq);
      });
  rtsp_client_->set_rx_timeout(10);  // 10 seconds timeout
}

int RtspSource::RtspNotifyCallback(int event) {
  LOGI(TAG, "RTSP Notify Event: %d", event);
  // Handle different RTSP events as needed
  switch (event) {
    case RTSP_EVE_CONNSUCC:  // Example event code for connection established
    {
      LOGI(TAG, "RTSP connection established");
      int codec = rtsp_client_->video_codec();
      if (codec == VIDEO_CODEC_H264) {
      } else if (codec == VIDEO_CODEC_H265) {
      } else {
        LOGE(TAG, "Unsupport codec");
      }
      // if (!initialized_.load()) {
      //   Init();
      // }
      break;
    }
    case RTSP_EVE_STOPPED:  // Example event code for connection lost
      LOGW(TAG, "RTSP connection lost");
      break;
    case RTSP_EVE_CONNFAIL:
      LOGW(TAG, "RTSP connection failed");
      break;
    case RTSP_EVE_NOSIGNAL:
      LOGW(TAG, "RTSP no signal");
      break;
    case RTSP_EVE_NODATA:
      LOGW(TAG, "RTSP no data");
      break;
    default:
      LOGW(TAG, "Unknown RTSP event: %d", event);
      break;
  }
  return 0;
}

int RtspSource::RtspVideoCallback(uint8* data, int len, uint32 ts, uint16 seq) {
  mpp_decoder_->PutPacket(data, len, 0);
  return 0;
}

void RtspSource::OnDecodedFrame(const VideoFrameSlot& frame) {
  if (frame.fd <= 0) {
    LOGE(TAG, "invalid fd");
    return;
  }

  video_width_ = frame.width;
  video_height_ = frame.height;

  {
    std::lock_guard<std::mutex> lock(sink_mutex_);

    current_frame_ = frame;
    for (auto& sink : sinks_) {
      sink->OnVideoFrame(current_frame_);
    }
  }
}

void RtspSource::AddVideoSink(VideoSink* sink) {
  std::lock_guard<std::mutex> lock(sink_mutex_);

  if (std::find(sinks_.begin(), sinks_.end(), sink) != sinks_.end()) {
    LOGW(TAG, "Sink already added");
    return;
  }

  sinks_.push_back(sink);
  LOGI(TAG, "Added video sink, total sinks: %zu", sinks_.size());
}

void RtspSource::RemoveVideoSink(VideoSink* sink) {
  std::lock_guard<std::mutex> lock(sink_mutex_);

  if (std::find(sinks_.begin(), sinks_.end(), sink) == sinks_.end()) {
    LOGW(TAG, "Sink not found");
    return;
  }

  sinks_.erase(std::remove(sinks_.begin(), sinks_.end(), sink), sinks_.end());
  LOGI(TAG, "Removed video sink, total sinks: %zu", sinks_.size());
}

bool RtspSource::Start() {
  if (!rtsp_client_) {
    LOGE(TAG, "RTSP client is not initialized");
    return false;
  }

  auto success = rtsp_client_->rtsp_start(url_.c_str(), nullptr, nullptr);
  if (!success) {
    LOGE(TAG, "Start rtsp client failed");
    return false;
  }

  LOGI(TAG, "RTSP client started with url: %s", url_.c_str());
  return true;
}

void RtspSource::Stop() {
  if (rtsp_client_) {
    rtsp_client_->rtsp_stop();
  }

  if (mpp_decoder_) {
    mpp_decoder_->PutPacket(nullptr, 0, 1);  // send EOS
  }

  LOGI(TAG, "RTSP client stopped");
}

}  // namespace rtsp