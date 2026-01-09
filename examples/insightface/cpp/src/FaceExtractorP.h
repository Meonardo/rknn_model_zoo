#ifndef FACE_EXTRACTOR_P_H
#define FACE_EXTRACTOR_P_H

#include <array>
#include <string>
#include <vector>

#include "utils/Utils.h"

namespace face {

class FaceExtractorP {
 public:
  FaceExtractorP(const std::string& model_path, int frame_width, int frame_height);
  FaceExtractorP(rknn_context* ctx, int frame_width, int frame_height);
  ~FaceExtractorP();

  int Extract(rga_buffer_t* cropped_face, float* lmk, Embeddings& embeddings);

  rknn_context GetRknnContext() const {
    return rknn_ctx_;
  }

 private:
  std::string model_path_;
  int frame_width_;
  int frame_height_;
  bool initialized_;

  rknn_context rknn_ctx_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  rknn_input_output_num io_num_;
  std::vector<rknn_tensor_mem*> input_mems_;
  std::vector<rknn_tensor_mem*> output_mems_;

  QntParam deqnt_params_;
  rga_buffer_t input_tensor_rga_buffer_;

  uint32_t model_input_width_;
  uint32_t model_input_height_;

  int Init(rknn_context* ctx);
  void DeInit();
  void CreateRgaBuffers();
  void DestroyRgaBuffers();

  rga_buffer_t* Affine(rga_buffer_t* cropped_face, cv::Mat& src, float* lmk);
};

} // namespace face

#endif // FACE_EXTRACTOR_P_H