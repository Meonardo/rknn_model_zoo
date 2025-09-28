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

#include "RgaBufferPool.h"
#include "RingBuffer.h"
#include "VideoCommon.h"
#include "im2d_type.h"
#include "rknn_api.h"

namespace det {

constexpr uint16_t kNumOfClasses = 7;

struct LetterBox {
  int x_pad;
  int y_pad;
  float scale;
};

struct DetectedObject {
  cv::Rect box;
  int label;
  float score;

  explicit DetectedObject(cv::Rect box_, int label_, float score_)
      : box(box_), label(label_), score(score_) {}
  ~DetectedObject() = default;
};

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
  explicit DetSource(std::string_view id);
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

  rknn_context rknn_ctx_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  rknn_input_output_num io_num_;
  std::vector<rknn_tensor_mem*> input_mems_;
  std::vector<rknn_tensor_mem*> output_mems_;

  // buffers
  std::unique_ptr<RingBuffer<VideoFrameSlot>> ring_buffer_;
  std::unique_ptr<RgaBufferPool> scale_buffer_pool_;
  std::unique_ptr<RgaBufferPool> rgb_buffer_pool_;
  std::unique_ptr<RgaBufferPool> nv12_buffer_pool_;
  std::unordered_map<int, std::unique_ptr<rga_buffer_t>> rga_buffers_;
  int last_success_fd_;
  std::atomic<bool> ready_;
  LetterBox letter_box_;
  uint32_t model_input_width_;
  uint32_t model_input_height_;
  uint32_t frame_width_;
  uint32_t frame_height_;
  std::vector<std::unique_ptr<OsdText>> osd_texts_;

  // video sinks
  std::vector<VideoSink*> sinks_;
  std::mutex sink_mutex_;
  VideoFrameSlot current_frame_;

  int Init();
  void DeInit();
  void CreateOsdTexts();
  void DestroyOsdTexts();

  static bool ImportRgaBuffer(const VideoFrameSlot& frame, rga_buffer_t* buffer);
  rga_buffer_t* Convert2RGBA(rga_buffer_t* src);
  rga_buffer_t* Convert2NV12(rga_buffer_t* src);
  rga_buffer_t* Letterbox(rga_buffer_t* src);
  rga_buffer_t* GetImportedRgaBuffer();
  void CalculateLetterBox();
  cv::Rect Unletterbox(float cx, float cy, float w, float h);
  cv::Rect UnletterboxV2(float x1, float y1, float x2, float y2);

  void MainLoop();

#if USE_QUANTIZED_MODEL
  std::vector<DetectedObject> PostProcess();
  int Processi8(int8_t* box_tensor, int32_t box_zp, float box_scale, int8_t* score_tensor,
                int32_t score_zp, float score_scale, int8_t* score_sum_tensor, int32_t score_sum_zp,
                float score_sum_scale, uint32_t grid_h, uint32_t grid_w, uint32_t stride,
                uint32_t dfl_len, std::vector<cv::Rect>& boxes, std::vector<float>& scores,
                std::vector<int>& class_id);
#else
  std::vector<DetectedObject> PostProcess();
#endif

  void DrawOsd(rga_buffer_t* buffer, const std::vector<DetectedObject>& objects);
};
}  // namespace det

#endif  // DET_SOURCE_H_