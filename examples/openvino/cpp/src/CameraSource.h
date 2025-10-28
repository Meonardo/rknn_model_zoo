//
// Created by Meonardo on 7/25/2024.
//

#ifndef GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_VIDEO_MIXER_CAMERASOURCE_H_
#define GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_VIDEO_MIXER_CAMERASOURCE_H_

#include <linux/videodev2.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "RgaBufferPool.h"
#include "VideoSink.h"
#include "utils/MppDecoder.h"

struct CameraFrame {
  void* start;
  size_t length;
  int export_fd;
  int sequence;
};

class CameraSource : public VideoSource {
 public:
  CameraSource(std::string_view id, std::string_view url);
  ~CameraSource() override;

  void AddVideoSink(VideoSink* sink) override;
  void RemoveVideoSink(VideoSink* sink) override;
  std::string GetId() const override {
    return id_;
  }
  VideoFrameSlot& GetVideoFrame() override;

 private:
  std::string id_;
  std::string device_id_;

  // video for linux 2
  int device_fd_;
  v4l2_buf_type buffer_type_;
  std::vector<CameraFrame> frames_;

  std::unique_ptr<std::thread> worker_;
  std::atomic<bool> capturing_;
  std::atomic<bool> running_;

  // video decoder
  v_dec::MppDecoder* mpp_decoder_;

  std::vector<VideoSink*> sinks_;
  std::mutex sink_mutex_;

  // resolution
  uint32_t max_width_;
  uint32_t max_height_;

  // rga
  std::unique_ptr<RgaBufferPool> nv12_buffer_pool_;
  std::unordered_map<int, std::unique_ptr<rga_buffer_t>> rga_buffers_;

  VideoFrameSlot current_frame_;

  void OpenCamera();
  void CloseCamera();
  bool CheckCameraAvailable();

  void Start();
  void Stop();
  void CaptureLoop();

  int RetrieveFrame();
  int PutFrame(int index);

  bool ImportRgaBuffer(int fd, rga_buffer_t* buffer) const;
  static bool Resize(rga_buffer_t* src, rga_buffer_t* dst);  
  
  // Decode callback
  void OnDecodedFrame(const VideoFrameSlot& frame);
};

#endif  // GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_VIDEO_MIXER_CAMERASOURCE_H_
