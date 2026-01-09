#ifndef DET_SOURCE_H_
#define DET_SOURCE_H_

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "FaceDetector.h"
#include "FaceExtractor.h"
#include "FaceExtractorP.h"
#include "RgaBufferPool.h"
#include "RingBuffer.h"
#include "VideoCommon.h"

#define PARALLEL_EXECUTION 1

#define MAX_PARALLEL_TASKS 4

namespace det {

class OsdText {
 public:
  explicit OsdText(const std::string& text);
  ~OsdText();

  int GetWidth() const {
    return w_;
  }
  int GetHeight() const {
    return h_;
  }

  const rga_buffer_t& GetRgaBuffer() const {
    return rga_buffer_;
  }

  const cv::Mat& GetRgbaMat() const {
    return rgba_;
  }

  const im_osd_t& GetOsdConfig() const {
    return config_;
  }

  im_osd_t* GetOsdConfigPtr() {
    return &config_;
  }

 private:
  std::string text_;
  int w_;
  int h_;
  size_t stride_;
  cv::Mat rgba_;

  rga_buffer_t rga_buffer_;
  im_osd_t config_;
};

class DetSource : public RawVideoSink, public VideoSource {
 public:
  explicit DetSource(std::string_view id, const std::vector<face::FaceLocation>& register_faces);
  ~DetSource() override;

  // Overrides
  void OnVideoFrameData(const void* buffer, size_t size) override {}
  void OnVideoFrame(const VideoFrameSlot& frame) override;

  void AddVideoSink(VideoSink* sink) override;
  void RemoveVideoSink(VideoSink* sink) override;
  [[nodiscard]] std::string GetId() const override {
    return id_;
  }
  VideoFrameSlot& GetVideoFrame() override {
    return current_frame_;
  }

  // Actions
  bool Start();
  void Stop();

 private:
  std::string id_;
  std::atomic<bool> running_;
  std::unique_ptr<std::thread> worker_thread_;

  // Detector & Extractor
  std::unique_ptr<face::FaceDetector> detector_;
#if PARALLEL_EXECUTION
  std::vector<rga_buffer_t*> crop_buffers_;
  std::vector<face::FaceExtractorP*> extractors_;
#else
  std::unique_ptr<face::FaceExtractor> extractor_;
#endif  // PARALLEL_EXECUTION

  // RGA buffers
  std::unique_ptr<RingBuffer<VideoFrameSlot>> ring_buffer_;
  std::unordered_map<int, std::unique_ptr<rga_buffer_t>> rga_buffers_;
  rga_buffer_t rgb_buffer_;
  rga_buffer_t output_nv12_buffer_;

  int last_success_fd_;
  std::atomic<bool> ready_;

  uint32_t frame_width_;
  uint32_t frame_height_;

  // video sinks
  std::vector<VideoSink*> sinks_;
  std::mutex sink_mutex_;
  VideoFrameSlot current_frame_;

  // OSD texts
  std::vector<face::FaceLocation> registered_faces_;
  std::vector<std::unique_ptr<OsdText>> osd_texts_;

  int Init();
  void DeInit();
  void CreateRgaBuffers();
  void DestroyRgaBuffers();

  rga_buffer_t* GetImportedRgaBuffer();
  static bool ImportRgaBuffer(const VideoFrameSlot& frame, rga_buffer_t* buffer);

  // Color conversion
  rga_buffer_t* Convert2RGB24(rga_buffer_t* src);
  rga_buffer_t* Convert2NV12(rga_buffer_t* src);

  void MainLoop();

  void CreateOsdTexts();
  void DestroyOsdTexts();

  void DrawOsd(rga_buffer_t* buffer, const std::vector<face::FaceLocation>& objects);

#if PARALLEL_EXECUTION
  void CreateCropBuffers();
  void DestroyCropBuffers();
  int ExtractEmbeddings(rga_buffer_t* src, std::vector<face::FaceLocation>& faces);
  int ExtractOne(face::FaceExtractorP* extractor, rga_buffer_t* src, rga_buffer_t* dst,
                 face::FaceLocation& face);
#endif  // PARALLEL_EXECUTION
};
}  // namespace det

#endif  // DET_SOURCE_H_