#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <string>
#include <vector>
#include <array>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "utils/Utils.h"

namespace face {

struct LetterBox {
  int x_pad;
  int y_pad;
  float scale;
};

class FaceDetector {
 public:
  explicit FaceDetector(const std::string& model_path, float det_threshold, float nms_threshold);
  ~FaceDetector();

  std::vector<FaceLocation> Detect(rga_buffer_t* input_buffer);

 private:
  std::string model_path_;
  bool initialized_;
  float det_threshold_;
  float nms_threshold_;

  rknn_context rknn_ctx_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  rknn_input_output_num io_num_;
  std::vector<rknn_tensor_mem*> input_mems_;
  std::vector<rknn_tensor_mem*> output_mems_;

  rga_buffer_t input_tensor_rga_buffer_;

  LetterBox letter_box_;
  uint32_t model_input_width_;
  uint32_t model_input_height_;
  uint32_t frame_width_;
  uint32_t frame_height_;

#if USE_QUANTIZED_MODEL
  std::vector<std::vector<QntParam>> deqnt_params_;
#endif

  // Cached anchors per stride index (0 -> stride 8, 1 -> 16, 2 -> 32)
  std::array<std::vector<float>, 3> anchors_cache_;

  int Init();
  void DeInit();
  void CreateRgaBuffers();
  void DestroyRgaBuffers();

  // Scale
  rga_buffer_t* Scale(rga_buffer_t* src);
  void CalculateScale(rga_buffer_t* input_buffer);
  cv::Rect UnScale(float x1, float y1, float x2, float y2);

  std::vector<FaceLocation> PostProcess();
  std::vector<FaceLocation> PostProcessV2();
#if USE_QUANTIZED_MODEL
  void DecodeOutput(const int8_t* cls_pred, const int8_t* box_pred, const int8_t* lmk_pred,
                    int stride, const std::vector<QntParam>& p, std::vector<FaceLocation>& results);
  void DecodeOutputV2(const int8_t* scores_raw, const int8_t* boxes_raw, const int8_t* lmk_raw,
                      int stride, const std::vector<QntParam>& p,
                      std::vector<FaceLocation>& results);
#else
  void DecodeOutput(const float* cls_pred, const float* box_pred, const float* lmk_pred, int stride,
                    std::vector<FaceLocation>& results);
#endif
};

}  // namespace face

#endif  // FACEDETECTOR_H