#ifndef FACEEXTRACTOR_H
#define FACEEXTRACTOR_H

#include <array>
#include <string>
#include <vector>

#include "utils/Utils.h"

namespace face {

class FaceExtractor {
 public:
  explicit FaceExtractor(const std::string& model_path, int frame_width, int frame_height);
  ~FaceExtractor();

  int Extract(rga_buffer_t* input_buffer, std::vector<FaceLocation>& faces);

 private:
  std::string model_path_;
  int frame_width_;
  int frame_height_;
  bool initialized_;
  bool full_size_crop_;

  rknn_context rknn_ctx_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  rknn_input_output_num io_num_;
  std::vector<rknn_tensor_mem*> input_mems_;
  std::vector<rknn_tensor_mem*> output_mems_;

  cv::Mat cropped_mat_;
  rga_buffer_t src_image_rga_buffer_;  // frame_widthxframe_height full size source image

  rga_buffer_t crop_image_rga_buffer_;    // 112x112 cropped face from the source image
  rga_buffer_t affine_image_rga_buffer_;  // 112x112 after affine transform

  rga_buffer_t input_tensor_rga_buffer_;

  uint32_t model_input_width_;
  uint32_t model_input_height_;

#if USE_QUANTIZED_MODEL
  QntParam deqnt_params_;
#endif

  int Init();
  void DeInit();
  void CreateRgaBuffers();
  void DestroyRgaBuffers();

  int ExtractFromFullSize(rga_buffer_t* input_buffer, std::vector<FaceLocation>& faces);

  rga_buffer_t* Crop(cv::Mat& src, FaceLocation& face);
  rga_buffer_t* Crop(rga_buffer_t* src, const cv::Rect2f& rect, float* lmk);
  rga_buffer_t* Affine(cv::Mat& src, float* lmk);
};

}  // namespace face

#endif  // FACEEXTRACTOR_H